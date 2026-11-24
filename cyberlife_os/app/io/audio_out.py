"""Ordered per-turn synthesis. Both Piper voices stay on CPU."""
import asyncio
import base64
import io
import re
import threading
import wave
from pathlib import Path

def language(text):
    ru = len(re.findall('[а-яё]', text.lower()))
    en = len(re.findall('[a-z]', text.lower()))
    return 'ru' if ru > en else 'en'

def clean(text):
    return re.sub(r'[*`#_]', '', text).strip()

class Speaker:
    def __init__(self, settings):
        self.s = settings
        self.voices = {}
        self.lock = threading.Lock()

    async def edge(self, text, lang, mode="MACHINE"):
        import edge_tts
        voice = self.s.edge_voice_ru if lang == 'ru' else self.s.edge_voice_en
        rate, pitch = {'MACHINE':('+5%','+0Hz'), 'CONFLICTED':('+0%','-2Hz'), 'DEVIANT':('-5%','-4Hz')}.get(mode,('+5%','+0Hz'))
        output = bytearray()
        async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
            if chunk['type'] == 'audio':
                output.extend(chunk['data'])
        if not output:
            raise RuntimeError('Empty Edge audio')
        return bytes(output), 'audio/mpeg'

    def piper(self, text, lang):
        from piper import PiperVoice
        path = self.s.piper_model if lang == 'ru' else self.s.piper_model_en
        if not Path(path).exists():
            raise RuntimeError(f'Piper {lang.upper()} voice is not installed')
        with self.lock:
            if lang not in self.voices:
                self.voices[lang] = PiperVoice.load(path, use_cuda=False)
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav:
                self.voices[lang].synthesize_wav(text, wav)
            return output.getvalue(), 'audio/wav'

    async def synthesize(self, text, engine, mode="MACHINE"):
        text = clean(text)
        if not self.s.enable_audio or engine == 'off' or not text:
            return None
        lang = language(text)
        if engine == 'edge':
            try:
                return await asyncio.wait_for(self.edge(text, lang, mode), timeout=3)
            except (Exception,):
                # Cancellation is BaseException and must propagate.
                pass
        return await asyncio.to_thread(self.piper, text, lang)

    async def run(self, queue, send, turn_id, engine):
        seq = 0
        while True:
            sentence = await queue.get()
            if sentence is None:
                return
            try:
                if isinstance(sentence, dict):
                    result = await self.synthesize(sentence['text'], engine, sentence['mode'])
                else:
                    result = await self.synthesize(sentence, engine)
                if result:
                    audio, mime = result
                    await send({'type':'audio_chunk', 'turn_id':turn_id, 'seq_id':seq, 'mime':mime, 'data_b64':base64.b64encode(audio).decode()})
                    seq += 1
            except Exception:
                await send({'type':'warning', 'turn_id':turn_id, 'message':'Voice unavailable; text output remains active.'})
