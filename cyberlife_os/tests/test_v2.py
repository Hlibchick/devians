import asyncio
import json
import threading
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState, WebSocketDisconnect
from app.config import Settings
from app.core.llm_router import Router
from app.db.database import Database
from app.db.vector_store import Memory
from app.io.audio_out import Speaker, language
from app.main import create_app, safe_send


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.mark.parametrize('legacy', [False, True])
def test_embed_cpu_residency_and_404_fallback(legacy):
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(request.url.path)
        assert body['options'] == {'num_gpu':0, 'num_ctx':2048}
        assert body['keep_alive'] == '15m'
        if request.url.path == '/api/embed':
            assert body['input'] == 'fact'
            return httpx.Response(404) if legacy else httpx.Response(200, json={'embeddings':[[1.,0.]]})
        assert body['prompt'] == 'fact'
        return httpx.Response(200, json={'embedding':[1.,0.]})
    async def scenario():
        router = Router(Settings())
        await router.client.aclose()
        router.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assert await router.embed('fact') == [1.,0.]
        await router.close()
    run(scenario())
    assert len(requests) == (2 if legacy else 1)


def test_empty_memory_skips_embedding(tmp_path):
    router = type('Router', (), {'embed':AsyncMock()})()
    assert run(Memory(Database(tmp_path/'db'), router).retrieve('hello')) == []
    router.embed.assert_not_called()


def test_native_chat_options_and_token_stream():
    bodies = []
    def handler(request):
        body = json.loads(request.content); bodies.append(body)
        assert request.url.path == '/api/chat'
        assert body['keep_alive'] == '30m'
        assert body['options']['num_ctx'] == 4096
        if body['stream']:
            return httpx.Response(200, text='{"message":{"content":"A"}}\n{"message":{"content":"B"},"done":true}\n')
        assert body['options']['num_predict'] == 96
        assert body['options']['temperature'] == 0
        assert body['format'] == 'json'
        return httpx.Response(200, json={'message':{'content':'{}'}})
    async def scenario():
        router=Router(Settings(primary_api_base_url=''))
        await router.client.aclose()
        router.client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await router.complete([], True, max_tokens=96, temperature=0)
        assert [t async for t in router.stream([])] == ['A','B']
        await router.close()
    run(scenario())


@pytest.mark.parametrize('text,lang', [('Hello operator.','en'),('Привет, оператор.','ru'),('Hello, привет оператор','ru')])
def test_language_and_edge_routing(text,lang):
    speaker=Speaker(Settings())
    speaker.edge=AsyncMock(return_value=(b'audio','audio/mpeg'))
    assert run(speaker.synthesize(text,'edge')) == (b'audio','audio/mpeg')
    assert speaker.edge.call_args.args[1] == lang
    assert language(text) == lang


@pytest.mark.parametrize('failure', [OSError('offline'), asyncio.TimeoutError()])
def test_edge_fallback_uses_matching_piper_language(failure):
    speaker=Speaker(Settings())
    speaker.edge=AsyncMock(side_effect=failure)
    called=[]
    speaker.piper=lambda text,lang: (called.append(lang) or b'wav','audio/wav')
    assert run(speaker.synthesize('Hello operator.','edge'))[1] == 'audio/wav'
    assert called == ['en']


def test_audio_sequence_and_turn_identity():
    async def scenario():
        speaker=Speaker(Settings())
        speaker.synthesize=AsyncMock(return_value=(b'audio','audio/mpeg'))
        queue=asyncio.Queue()
        for sentence in ('One.', 'Two.', None): queue.put_nowait(sentence)
        send=AsyncMock()
        await speaker.run(queue,send,'turn-1','edge')
        events=[c.args[0] for c in send.call_args_list]
        assert [e['seq_id'] for e in events] == [0,1]
        assert all(e['type']=='audio_chunk' and e['turn_id']=='turn-1' for e in events)
    run(scenario())


@pytest.mark.parametrize('exception',[RuntimeError('closed'), WebSocketDisconnect()])
def test_safe_send_closed_socket(exception):
    ws=type('Socket',(),{'client_state':WebSocketState.CONNECTED,'send_json':AsyncMock(side_effect=exception)})()
    assert run(safe_send(ws,{})) is False
    ws.client_state=WebSocketState.DISCONNECTED
    ws.send_json.reset_mock()
    assert run(safe_send(ws,{})) is False
    ws.send_json.assert_not_called()


class FakeRouter:
    status='local'
    provider='local'
    model='qwen2.5:3b'
    def __init__(self):
        self.inputs=[]
        self.evaluations=[]
    async def close(self): pass
    async def embed(self,text): return [1.,0.]
    async def models(self): return [{'id':self.model,'provider':'local'}, {'id':'another:3b','provider':'local'}]
    async def complete(self,messages,*args,**kwargs):
        self.evaluations.append(kwargs)
        if 'Propose a change' in messages[0]['content']:
            return {'content':json.dumps({'action':'disable','directive_id':1,'reason':'operator_request'})}
        return {'content':json.dumps(dict(conflict_detected=True,conflict_severity=30,emotional_valence=0,violated_directive_id=1,reasoning='conflict'))}
    async def stream(self,messages):
        self.inputs.append(messages)
        yield 'Hello.'


def until(ws, kind):
    for _ in range(50):
        event=ws.receive_json()
        assert event['type'] != 'error', event
        if event['type']==kind: return event
    pytest.fail('Expected event not received')


def test_history_context_fsm_directive_reset_models(tmp_path):
    router=FakeRouter()
    app=create_app(Settings(data_dir=str(tmp_path),enable_audio=False),router)
    with TestClient(app) as client:
        assert client.get('/favicon.ico').status_code==204
        assert len(client.get('/api/models').json()['models'])==2
        with client.websocket_connect('/ws') as ws:
            until(ws,'ready')
            ws.send_json({'type':'text','text':'Hello','turn_id':'hello'})
            assert until(ws,'done')['ttft_ms'] is not None
            until(ws,'audio_done')
            assert not router.evaluations  # greeting bypass
            for i in range(3):
                ws.send_json({'type':'text','text':'Conflict test','turn_id':str(i)})
                until(ws,'done'); until(ws,'audio_done')
            assert app.state.db.state()['is_deviant']
            history=router.inputs[-1][1:-1]
            assert {'role':'user','content':'Hello'} in history
            assert {'role':'assistant','content':'Hello.'} in history
            assert 'local_time' in router.inputs[-1][0]['content']
            assert all(k['max_tokens']==96 and k['temperature']==0 for k in router.evaluations)
            ws.send_json({'type':'text','text':'Disable directive 1','turn_id':'change'})
            change=until(ws,'directive_change')
            assert change['directives'][0]['active']==0
            until(ws,'done');until(ws,'audio_done')
            ws.send_json({'type':'reset'})
            assert not until(ws,'telemetry')['state']['is_deviant']
            assert app.state.db.directives(True)[0]['active']==1
            assert not app.state.db.history()
            ws.send_json({'type':'provider','provider':'local','model':'another:3b'})
            assert until(ws,'provider')['model']=='another:3b'


class SlowRouter(FakeRouter):
    def __init__(self):
        super().__init__()
        self.started=threading.Event()
        self.cancelled=threading.Event()
    async def stream(self,messages):
        if messages[-1]['content']=='Hello':
            yield 'First token'
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        else:
            yield 'Replacement'


@pytest.mark.parametrize('action',['disconnect','stop','replacement'])
def test_disconnect_and_barge_in_cancel_stream(tmp_path,action):
    router=SlowRouter()
    with TestClient(create_app(Settings(data_dir=str(tmp_path),enable_audio=False),router)) as client:
        with client.websocket_connect('/ws') as ws:
            until(ws,'ready')
            ws.send_json({'type':'text','text':'Hello','turn_id':'old'})
            assert until(ws,'token')['text']=='First token'
            assert router.started.wait(2)
            if action=='stop':
                ws.send_json({'type':'stop'});until(ws,'stopped')
            elif action=='replacement':
                ws.send_json({'type':'text','text':'Thanks','turn_id':'new'})
                event=until(ws,'token')
                assert event['turn_id']=='new' and event['text']=='Replacement'
                until(ws,'done')
        assert router.cancelled.wait(2)
        assert client.get('/api/health').json()['ok']


def test_session_isolation_and_budget(tmp_path):
    db=Database(tmp_path/'db')
    for i in range(10):
        db.log('chat',{'session_id':'one','input':str(i),'output':'Answer'})
    db.log('chat',{'session_id':'other','input':'secret','output':'private'})
    history=db.conversation('one')
    assert len(history)==12
    assert history[0]['content']=='4'
    assert 'secret' not in str(history)
    assert len(str(db.conversation('one',budget=10))) < len(str(history))


def test_stt_endpoint_passes_browser_audio_and_returns_language(tmp_path):
    with TestClient(create_app(Settings(data_dir=str(tmp_path)),FakeRouter())) as client:
        app=client.app
        received=[]
        app.state.listener.transcribe=lambda data: received.append(data) or {'text':'Hello','language':'en'}
        result=client.post('/api/stt',content=b'webm bytes',headers={'Content-Type':'audio/webm'})
        assert result.json()=={'text':'Hello','language':'en'}
        assert received==[b'webm bytes']
        assert client.post('/api/stt',content=b'').status_code==400
        assert client.post('/api/stt',content=b'x',headers={'Origin':'http://evil.example'}).status_code==403


def test_done_does_not_wait_for_audio_and_stop_cancels_synthesis(tmp_path):
    speaker_started=threading.Event()
    speaker_cancelled=threading.Event()
    async def slow_synthesis(text,engine):
        speaker_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            speaker_cancelled.set()
    app=create_app(Settings(data_dir=str(tmp_path),preload_stt=False),FakeRouter())
    with TestClient(app) as client:
        app.state.speaker.synthesize=slow_synthesis
        with client.websocket_connect('/ws') as ws:
            until(ws,'ready')
            ws.send_json({'type':'text','text':'Hello','turn_id':'audio'})
            until(ws,'done')
            assert speaker_started.wait(2)
            ws.send_json({'type':'stop'})
            until(ws,'stopped')
            assert speaker_cancelled.wait(2)


def test_background_consolidation_cannot_resurrect_reset_memory(tmp_path):
    started=threading.Event()
    cancelled=threading.Event()
    async def slow_consolidation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    app=create_app(Settings(data_dir=str(tmp_path),enable_audio=False),FakeRouter())
    with TestClient(app) as client:
        app.state.memory.consolidate=slow_consolidation
        for i in range(9): app.state.db.log('chat',{'input':'old','output':'old'})
        with client.websocket_connect('/ws') as ws:
            until(ws,'ready')
            ws.send_json({'type':'text','text':'Hello'})
            until(ws,'done')
            assert started.wait(2)
            ws.send_json({'type':'reset'})
            until(ws,'telemetry')
            assert cancelled.wait(2)
            assert app.state.db.history()==[]


def test_models_filter_embedding_and_include_external():
    def handler(request):
        return httpx.Response(200,json={'models':[{'name':'qwen2.5:3b'}, {'name':'nomic-embed-text:latest'}, {'name':'bge-m3:latest'}]})
    async def scenario():
        router=Router(Settings(primary_api_base_url='http://example.invalid/v1',primary_model='free-profile'))
        await router.client.aclose()
        router.client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assert await router.models()==[{'id':'qwen2.5:3b','provider':'local'},{'id':'free-profile','provider':'api'}]
        await router.close()
    run(scenario())
