import json
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app
class Fake:
    status='mock'
    provider='api'
    async def close(self): pass
    async def embed(self,text): return [1.,0.,0.]
    async def complete(self,*args,**kwargs): return {'content':json.dumps(dict(conflict_detected=True,conflict_severity=30,emotional_valence=0,violated_directive_id=1,reasoning='test'))}
    async def stream(self,messages):
        yield 'Hello.'

def test_http_websocket(tmp_path):
    app=create_app(Settings(data_dir=str(tmp_path),enable_audio=False,enable_vision=False),Fake())
    with TestClient(app) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/health').json()['ok']
        with client.websocket_connect('/ws') as ws:
            assert ws.receive_json()['type']=='telemetry';ws.receive_json()
            for _ in range(3):
                ws.send_json({'type':'text','text':'Test'})
                messages=[]
                while True:
                    msg=ws.receive_json();messages.append(msg)
                    if msg['type']=='done': break
                    assert msg['type']!='error'
                assert any(m.get('text')=='Hello.' for m in messages)
            assert client.get('/api/health').json()['state']['is_deviant']
            ws.send_json({'type':'reset'})
            while (event := ws.receive_json())['type'] != 'telemetry':
                assert event['type'] != 'error'
            assert not event['state']['is_deviant']

def test_vector_search(tmp_path):
    import asyncio,sqlite_vec
    from app.db.database import Database
    from app.db.vector_store import Memory
    db=Database(tmp_path/'v.db')
    with db.connection() as c:
        c.execute('INSERT INTO facts(text,embedding) VALUES(?,?)',('fact',sqlite_vec.serialize_float32([1.,0.,0.])))
    assert asyncio.run(Memory(db,Fake()).retrieve('query'))[0]['text']=='fact'

def test_unavailable_models_degrade_to_mock(tmp_path):
    settings=Settings(data_dir=str(tmp_path),enable_audio=False,enable_vision=False,ollama_base_url='http://127.0.0.1:1',primary_api_base_url='http://127.0.0.1:1/v1')
    with TestClient(create_app(settings)) as client:
        with client.websocket_connect('/ws') as ws:
            ws.receive_json();ws.receive_json()
            ws.send_json({'type':'text','text':'Hello'})
            tokens=[]
            while True:
                msg=ws.receive_json()
                assert msg['type']!='error'
                if msg['type']=='token': tokens.append(msg['text'])
                if msg['type']=='done':
                    assert msg['provider']=='mock';break
            assert '[MOCK]' in ''.join(tokens)
