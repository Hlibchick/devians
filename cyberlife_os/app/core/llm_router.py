"""Ollama native chat keeps runtime options explicit; API uses SSE."""
import json
import httpx

class Router:
    def __init__(self, settings):
        self.s = settings
        self.provider = 'api' if settings.primary_api_base_url else 'local'
        self.model = settings.local_fallback_model
        self.status = 'unprobed'
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
            if vision and label == 'local':
                continue
            try:
                body = self.body(self.s.vision_model if vision else model, label, messages, False, json_mode, max_tokens, temperature, tools)
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

    async def stream(self, messages):
        for url, headers, model, label in self.targets():
            emitted = False
            try:
                async with self.client.stream('POST', url, headers=headers, json=self.body(model, label, messages, True)) as result:
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
                    models.append({'id': name, 'provider': 'local'})
        except (httpx.HTTPError, ValueError):
            pass
        if self.s.primary_api_base_url:
            models.append({'id': self.s.primary_model, 'provider': 'api'})
        return models
