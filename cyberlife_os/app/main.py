import asyncio
import logging
import httpx
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from uvicorn.protocols.utils import ClientDisconnected
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.config import Settings
from app.db.database import Database
from app.db.vector_store import Memory
from app.core.llm_router import Router
from app.core.graph import create_graph
from app.io.audio_out import Speaker
from app.io.audio_in import Listener, STTError
from app.io.vision import capture

log = logging.getLogger(__name__)

async def safe_send(ws, payload):
    if ws.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await ws.send_json(payload)
        return True
    except (WebSocketDisconnect, ClientDisconnected, RuntimeError):
        return False

async def cancel(task):
    if task:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def create_app(settings=None, router=None):
    settings = settings or Settings()
    @asynccontextmanager
    async def lifespan(app):
        app.state.db = Database(Path(settings.data_dir) / 'cyberlife.db')
        app.state.router = router or Router(settings)
        app.state.lock = asyncio.Lock()
        app.state.memory_lock = asyncio.Lock()
        app.state.background = set()
        app.state.turns = set()
        app.state.memory = Memory(app.state.db, app.state.router)
        app.state.speaker = Speaker(settings)
        app.state.listener = Listener()
        if settings.enable_audio and settings.preload_stt:
            try:
                await asyncio.to_thread(app.state.listener.load)
            except Exception:
                log.exception('STT preload unavailable; text mode remains active')
        async def consolidate():
            async with app.state.lock:
                async with app.state.memory_lock:
                    await app.state.memory.consolidate()
        scheduler = AsyncIOScheduler()
        scheduler.add_job(consolidate, 'interval', minutes=5, max_instances=1)
        scheduler.start()
        yield
        scheduler.shutdown(wait=False)
        for task in list(app.state.turns | app.state.background):
            await cancel(task)
        await app.state.router.close()

    app = FastAPI(lifespan=lifespan)

    def spawn(coroutine):
        task = asyncio.create_task(coroutine)
        app.state.background.add(task)
        def finished(task):
            app.state.background.discard(task)
            if not task.cancelled() and task.exception():
                log.error('Background memory operation failed: %s', type(task.exception()).__name__)
        task.add_done_callback(finished)
        return task

    async def persist(result):
        app.state.db.log('chat', {'session_id':result['session_id'], 'input':result['text'], 'output':result['response'], 'kind':result.get('kind','text'), 'timestamp':result['timestamp']})
        with app.state.db.connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM logs WHERE kind='chat'").fetchone()[0]
        if count >= 10:
            async with app.state.lock:
                async with app.state.memory_lock:
                    await app.state.memory.consolidate()

    @app.get('/api/health')
    async def health():
        return {'ok':True, 'provider':app.state.router.status, 'state':app.state.db.state()}

    @app.get('/favicon.ico')
    async def favicon():
        return Response(status_code=204)

    @app.get('/api/models')
    async def models():
        return {'models':await app.state.router.models(), 'preferred_model':settings.local_fallback_model, 'active':{'provider':app.state.router.provider, 'model':app.state.router.model if app.state.router.provider == 'local' else settings.primary_model}}

    @app.post('/api/stt')
    async def stt(request: Request):
        if not settings.enable_audio:
            raise HTTPException(409, 'Audio disabled')
        origin = request.headers.get('origin')
        if origin and origin.split('://',1)[-1] != request.headers.get('host'):
            raise HTTPException(403, 'Origin mismatch')
        audio = bytearray()
        async for chunk in request.stream():
            audio.extend(chunk)
            if len(audio) > 10 * 1024 * 1024:
                raise HTTPException(413, 'Audio exceeds 10 MB')
        if not audio:
            raise HTTPException(400, 'Empty audio')
        try:
            return await asyncio.to_thread(app.state.listener.transcribe, bytes(audio))
        except STTError as exc:
            log.exception('STT failed: stage=%s bytes=%d content_type=%s', exc.stage, len(audio), request.headers.get('content-type', '')[:100])
            raise HTTPException(exc.status_code, f'STT [{exc.stage}]: {exc}') from None
        except Exception:
            log.exception('STT failed: unexpected error, bytes=%d', len(audio))
            raise HTTPException(500, 'STT [unexpected]: see the server traceback; text mode remains available.') from None

    @app.websocket('/ws')
    async def websocket(ws: WebSocket):
        origin = ws.headers.get('origin')
        if origin and origin.split('://',1)[-1] != ws.headers.get('host'):
            await ws.close(code=1008)
            return
        await ws.accept()
        session_id = str(uuid.uuid4())
        active = None
        engine = settings.tts_engine
        sendlock = asyncio.Lock()
        async def send(value):
            async with sendlock:
                return await safe_send(ws, value)
        await send({'type':'telemetry', 'state':app.state.db.state(), 'directives':app.state.db.directives(True)})
        await send({'type':'ready', 'tts_engine':engine, 'session_id':session_id})

        async def turn(event, turn_id, selected_engine):
            started = time.perf_counter()
            ttft = None
            model_ttft = None
            queue = asyncio.Queue()
            worker = None
            async def emit(payload):
                nonlocal ttft, model_ttft
                payload['turn_id'] = turn_id
                if payload['type'] == 'model_token':
                    if model_ttft is None:
                        model_ttft = (time.perf_counter()-started)*1000
                        await send({'type':'model_timing','turn_id':turn_id,'model_ttft_ms':round(model_ttft,1)})
                    return
                if payload['type'] == 'token' and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                    payload['ttft_ms'] = round(ttft, 1)
                if not await send(payload):
                    raise asyncio.CancelledError()
            try:
                async with app.state.lock:
                    kind = event.get('type', 'text')
                    text = event.get('text','')
                    if kind == 'voice':
                        if not settings.server_microphone or not settings.enable_audio:
                            raise ValueError('Server microphone disabled; use browser recording')
                        transcript = await asyncio.to_thread(app.state.listener.server_microphone)
                        text = transcript['text']
                        await emit({'type':'transcript', **transcript})
                    elif kind == 'vision_event':
                        if not settings.enable_vision:
                            raise ValueError('Vision disabled')
                        frame = await asyncio.to_thread(capture)
                        result = await app.state.router.complete([{'role':'user','content':[{'type':'text','text':'Describe this image.'},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+frame}}]}], vision=True)
                        text = '[VISUAL INPUT]: ' + (result.get('content') or 'Vision unavailable / Анализ изображения недоступен.')
                    elif kind != 'text':
                        raise ValueError('Unknown event')
                    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
                        raise ValueError('Input must contain 1–16000 characters')
                    if hasattr(app.state.router, 'ensure_gpu'):
                        try:
                            await app.state.router.ensure_gpu(app.state.router.model)
                        except httpx.HTTPError:
                            pass  # Router retains diagnostic/API fallback when Ollama is offline.
                    worker = asyncio.create_task(app.state.speaker.run(queue, emit, turn_id, selected_engine))
                    graph = create_graph(app.state.db, app.state.router, app.state.memory, app.state.speaker, emit, session_id=session_id, speech_queue=queue)
                    result = await graph.ainvoke({'text':text, 'kind':kind})
                    await emit({'type':'done', 'provider':app.state.router.status, 'ttft_ms':round(ttft,1) if ttft is not None else None, 'model_ttft_ms':round(model_ttft,1) if model_ttft is not None else None, 'total_ms':round((time.perf_counter()-started)*1000,1)})
                    # Scheduling happens after completion delivery. No embedding work blocks done.
                    spawn(persist(result))
                    queue.put_nowait(None)
                await worker
                await emit({'type':'audio_done'})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await send({'type':'error', 'turn_id':turn_id, 'message':str(exc)[:200], 'input_type':event.get('type')})
            finally:
                await cancel(worker)

        try:
            while True:
                try:
                    event = await ws.receive_json()
                    if not isinstance(event, dict):
                        raise ValueError('Expected JSON object')
                    kind = event.get('type','text')
                    if kind in ('stop','reset','provider','tts','text','voice','vision_event'):
                        await cancel(active)
                        active = None
                    if kind == 'stop':
                        await send({'type':'stopped'})
                    elif kind == 'reset' or event.get('text') == '/reset_memory':
                        # Reset invalidates every pending write/turn, including other sockets.
                        for task in list(app.state.turns | app.state.background):
                            await cancel(task)
                        async with app.state.memory_lock:
                            app.state.db.reset()
                        await send({'type':'telemetry','state':app.state.db.state(),'directives':app.state.db.directives(True),'memories':[]})
                    elif kind == 'tts':
                        if event.get('engine') not in ('edge','piper','off'):
                            raise ValueError('Invalid TTS engine')
                        engine = event['engine']
                        await send({'type':'tts', 'engine':engine})
                    elif kind == 'provider':
                        provider = event.get('provider')
                        if provider not in ('api','local'):
                            raise ValueError('Invalid provider')
                        available = await app.state.router.models()
                        model = event.get('model')
                        if not any(m['provider']==provider and m['id']==model for m in available):
                            raise ValueError('Model is not available')
                        async with app.state.lock:
                            app.state.router.provider = provider
                            if provider == 'local':
                                app.state.router.model = model
                        await send({'type':'provider', 'provider':provider, 'model':model})
                    elif kind in ('text','voice','vision_event'):
                        turn_id = str(event.get('turn_id') or uuid.uuid4())[:80]
                        active = asyncio.create_task(turn(event, turn_id, engine))
                        app.state.turns.add(active)
                        active.add_done_callback(app.state.turns.discard)
                    else:
                        raise ValueError('Unknown event')
                except ValueError as exc:
                    await send({'type':'error', 'message':str(exc)[:200]})
        except (WebSocketDisconnect, ClientDisconnected):
            pass
        finally:
            await cancel(active)

    app.mount('/', StaticFiles(directory=Path(__file__).parent/'static', html=True), name='static')
    return app

app = create_app()
