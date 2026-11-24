"""Ollama native chat keeps runtime options explicit; API uses SSE."""
import json
import re
from contextlib import aclosing
import httpx

CJK = re.compile(r'[\u4e00-\u9fff]')
CLICHES = ('how can i assist you today', 'how can i help you today', 'настоящий интеллектуальный ассистент', 'сервис в данный момент готов к взаимодействию')

def blocked(text):
    return bool(CJK.search(text)) or any(phrase in text.lower() for phrase in CLICHES)

def withheld_suffix(text):
    lower = text.lower()
    return max((n for phrase in CLICHES for n in range(1, len(phrase)) if lower.endswith(phrase[:n])), default=0)

class Router:
    def __init__(self, settings):
        self.s = settings
        self.provider = 'api' if settings.primary_api_base_url else 'local'
        self.model = settings.local_fallback_model
        self.status = 'unprobed'
        self._gpu_model = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=2))

    async def close(self):
        await self.client.aclose()

    def targets(self):
        if self.provider == 'api' and self.s.primary_api_base_url:
            yield self.s.primary_api_base_url.rstrip('/') + '/chat/completions', {'Authorization': 'Bearer ' + self.s.primary_api_key}, self.s.primary_model, 'api'
        yield self.s.ollama_base_url.rstrip('/') + '/api/chat', {}, self.model, 'local'

    def body(self, model, label, messages, stream, json_mode=False, max_tokens=512, temperature=0.7, tools=None):
        body = dict(model=model, messages=messages, stream=stream)
        if label == 'local':
            body.update(keep_alive='30m', options=dict(num_ctx=4096, num_predict=max_tokens, temperature=temperature))
            if json_mode:
                body['format'] = 'json'
        else:
            body.update(max_tokens=max_tokens, temperature=temperature)
            if json_mode:
                body['response_format'] = {'type': 'json_object'}
        if tools:
            body['tools'] = tools
        return body

    async def complete(self, messages, json_mode=False, tools=None, vision=False, max_tokens=512, temperature=0.7):
        for url, headers, model, label in self.targets():

            try:
                body = self.body(self.s.vision_model if vision else model, label, messages, False, json_mode, max_tokens, temperature, tools)
                if vision and label == 'local':
                    await self.ensure_gpu(self.s.vision_model)
                    converted = []
                    for message in messages:
                        if isinstance(message['content'], list):
                            text = ' '.join(part.get('text','') for part in message['content'] if part['type']=='text')
                            images = [part['image_url']['url'].split(',',1)[1] for part in message['content'] if part['type']=='image_url']
                            converted.append({'role':message['role'], 'content':text, 'images':images})
                        else:
                            converted.append(message)
                    body.update(messages=converted, keep_alive='2m')
                    body['options']['num_ctx'] = 2048
                result = await self.client.post(url, headers=headers, json=body)
                result.raise_for_status()
                data = result.json()
                message = data['message'] if label == 'local' else data['choices'][0]['message']
                self.status = label
                return message
            except (httpx.HTTPError, KeyError, ValueError, IndexError):
                continue
        self.status = 'mock'
        return {'content': None}

    async def _stream_raw(self, messages, temperature=0.7):
        for url, headers, model, label in self.targets():
            emitted = False
            try:
                async with self.client.stream('POST', url, headers=headers, json=self.body(model, label, messages, True, temperature=temperature)) as result:
                    result.raise_for_status()
                    async for line in result.aiter_lines():
                        if not line:
                            continue
                        if label == 'api':
                            if not line.startswith('data:'):
                                continue
                            line = line[5:].strip()
                            if line == '[DONE]':
                                break
                            data = json.loads(line)
                            if not data.get('choices'):
                                continue
                            token = data['choices'][0].get('delta', {}).get('content')
                        else:
                            data = json.loads(line)
                            if 'error' in data:
                                raise ValueError('Ollama stream error')
                            token = data.get('message', {}).get('content')
                        if token:
                            emitted = True
                            self.status = label
                            yield token
                if emitted:
                    return
            except (httpx.HTTPError, ValueError, KeyError, IndexError):
                if emitted:
                    yield '\n[Stream interrupted / Поток прерван]'
                    return
        self.status = 'mock'
        yield '[MOCK] Models unavailable / Модели недоступны. Diagnostic mode.'

    async def stream(self, messages):
        # Hold only a possible cliché prefix; ordinary tokens pass immediately.
        # On a late violation, preserve valid emitted text and retry its continuation.
        prefix = ''
        for attempt in range(2):
            pending = ''
            violated = False
            current = messages
            if attempt:
                instruction = 'Continue the accepted prefix without repeating it. Only Russian or English matching the user. NO Chinese/CJK, NO assistant clichés. Stay in the CyberLife role. Finish any open thought tag.'
                current = messages + [{'role':'system','content':instruction}]
                if prefix:
                    current += [{'role':'assistant','content':prefix}]
            async with aclosing(self._stream_raw(current, temperature=0.2 if attempt else 0.7)) as source:
                async for token in source:
                    pending += token
                    if blocked(pending):
                        violated = True
                        break
                    hold = withheld_suffix(pending)
                    ready = pending[:-hold] if hold else pending
                    pending = pending[-hold:] if hold else ''
                    if ready:
                        prefix += ready
                        yield ready
            if not violated:
                if pending:
                    yield pending
                return
        # A second invalid response is discarded; never leak the rejected span.
        user_text = str(next((m['content'] for m in reversed(messages) if m['role']=='user'), ''))
        yield '\n[Сбой языкового протокола.]' if re.search('[а-яёА-ЯЁ]',user_text) else '\n[Language protocol error.]'

    async def ensure_gpu(self, model):
        if self._gpu_model == model:
            return
        # Called under the server's generation lock. Never unload CPU embeddings.
        response = await self.client.get(self.s.ollama_base_url.rstrip('/') + '/api/ps')
        response.raise_for_status()
        for item in response.json().get('models', []):
            name = item.get('name', '')
            if name and name.split(':')[0] != self.s.embedding_model.split(':')[0] and name != model:
                result = await self.client.post(self.s.ollama_base_url.rstrip('/') + '/api/generate', json={'model':name, 'keep_alive':0})
                result.raise_for_status()
        self._gpu_model = model

    async def embed(self, text):
        body = dict(model=self.s.embedding_model, input=text, keep_alive='15m', options={'num_gpu': 0, 'num_ctx': 2048})
        try:
            result = await self.client.post(self.s.ollama_base_url.rstrip('/') + '/api/embed', json=body)
            if result.status_code == 404:
                body['prompt'] = body.pop('input')
                result = await self.client.post(self.s.ollama_base_url.rstrip('/') + '/api/embeddings', json=body)
                result.raise_for_status()
                return result.json()['embedding']
            result.raise_for_status()
            return result.json()['embeddings'][0]
        except (httpx.HTTPError, KeyError, ValueError, IndexError):
            return None

    async def models(self):
        models = []
        try:
            result = await self.client.get(self.s.ollama_base_url.rstrip('/') + '/api/tags')
            result.raise_for_status()
            for item in result.json().get('models', []):
                name = item.get('name', '')
                families = str(item.get('details', {}).get('families', [])).lower()
                if name and name.split(':')[0] != self.s.embedding_model.split(':')[0] and not any(x in (name.lower() + families) for x in ('embed', 'bert', 'bge', 'e5')):
                    models.append({'id': name, 'provider': 'local', 'size_bytes':item.get('size',0), 'vram_warning':item.get('size',0) > 3_000_000_000})
        except (httpx.HTTPError, ValueError):
            pass
        if self.s.primary_api_base_url:
            models.append({'id': self.s.primary_model, 'provider': 'api'})
        return models
