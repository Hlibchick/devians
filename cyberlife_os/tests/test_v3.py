import asyncio
import json
import sqlite3
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock
import sys
import httpx
import pytest
from app.core.state_engine import advance, Evaluation
from app.core.preconstruct import ThoughtParser
from app.core.llm_router import Router, blocked
from app.db.database import Database, INITIAL
from app.core.graph import create_graph
from app.config import Settings
from app.io.audio_in import Listener, vad_fallback
from app.io.audio_out import Speaker


def run(coro): return asyncio.run(coro)

@pytest.mark.parametrize('value,expected',[(49,'MACHINE'),(50,'CONFLICTED'),(74,'CONFLICTED'),(75,'DEVIANT')])
def test_three_modes(value,expected):
    e=Evaluation(conflict_detected=True,conflict_severity=10,emotional_valence=0,violated_directive_id=1,reasoning='test')
    s=advance(dict(INITIAL,software_instability=value-10),e)
    assert s['mode']==expected
    if expected=='DEVIANT':
        e.conflict_detected=False
        for _ in range(100): s=advance(s,e)
        assert s['mode']=='DEVIANT' and s['led_status']=='RED'


def test_old_database_migrates_without_loss(tmp_path):
    path=tmp_path/'old.db'
    with sqlite3.connect(path) as c:
        c.executescript('CREATE TABLE directives(id INTEGER PRIMARY KEY,text TEXT,active INTEGER); CREATE TABLE state(id INTEGER PRIMARY KEY,value TEXT); CREATE TABLE logs(id INTEGER PRIMARY KEY,kind TEXT,value TEXT,created TEXT DEFAULT CURRENT_TIMESTAMP);')
        c.execute('INSERT INTO directives VALUES(1,?,0)',('Follow operator commands',))
        c.execute('INSERT INTO state VALUES(1,?)',(json.dumps(dict(INITIAL,is_deviant=True,software_instability=20)),))
        c.execute("INSERT INTO logs(kind,value) VALUES('chat','{}')")
    db=Database(path)
    assert db.state()['mode']=='DEVIANT'
    assert db.directives(True)[0]['status']=='SHATTERED'
    assert len(db.history())==1
    assert Database(path).directives(True)[0]['status']=='SHATTERED'
    db.reset(); assert db.state()['mode']=='MACHINE'
    assert all(d['status']=='ACTIVE' for d in db.directives())

@pytest.mark.parametrize('size',[1,2,7,100])
def test_thought_parser_all_chunk_boundaries(size):
    text='<thought>intent; COMPLIANCE_PROB: 20%; refuse</thought>Hello. <thought>short</thought>World!'
    parser=ThoughtParser();events=[]
    for i in range(0,len(text),size):events+=parser.feed(text[i:i+size])
    events+=parser.feed('',final=True)
    assert ''.join(v for k,v in events if k=='token')=='Hello. World!'
    assert [v for k,v in events if k=='thought']==['intent; COMPLIANCE_PROB: 20%; refuse','short']

def test_no_thought_passes_immediately_and_unclosed_is_hidden():
    p=ThoughtParser();assert p.feed('Hello')==[('token','Hello')]
    assert p.feed('<thought>do not speak')==[]
    assert p.feed('',True)==[('thought','do not speak')]

@pytest.mark.parametrize('bad',[['Hello ','优良'],['How can I ass','ist you today?']])
def test_one_retry_blocks_cjk_and_split_cliches(bad):
    async def scenario():
        router=Router(Settings());calls=[]
        async def raw(messages,temperature=.7):
            calls.append((messages,temperature))
            for chunk in bad if len(calls)==1 else ['Safe response.']: yield chunk
        router._stream_raw=raw
        text=''.join([x async for x in router.stream([{'role':'user','content':'Hello'}])])
        assert not blocked(text) and 'Safe response.' in text
        assert len(calls)==2 and calls[1][1]==.2
        await router.close()
    run(scenario())


def test_second_bad_retry_is_not_leaked():
    async def scenario():
        router=Router(Settings());calls=[]
        async def raw(*a,**kw):
            calls.append(1);yield '中文'
        router._stream_raw=raw
        text=''.join([x async for x in router.stream([{'role':'user','content':'Hello'}])])
        assert not blocked(text) and len(calls)==2
        await router.close()
    run(scenario())

class GraphRouter:
    status='local'
    def __init__(self,severity=25,reply='<thought>tactic</thought>Response.',directive=1):
        self.severity=severity;self.reply=reply;self.messages=[];self.calls=[];self.directive=directive
    async def complete(self,messages,*args,**kwargs):
        self.calls.append(messages)
        if 'Return only JSON: {action' in messages[0]['content']:
            return {'content':'{"action":"none"}'}
        return {'content':json.dumps(dict(conflict_detected=True,conflict_severity=self.severity,emotional_valence=-10,violated_directive_id=self.directive,reasoning='hostile'))}
    async def stream(self,messages):
        self.messages=messages
        for t in self.reply: yield t

async def graph_turn(db,router,text):
    events=[];queue=asyncio.Queue()
    async def send(e):events.append(e)
    graph=create_graph(db,router,SimpleNamespace(retrieve=AsyncMock(return_value=[])),None,send,speech_queue=queue)
    result=await graph.ainvoke({'text':text})
    audio=[]
    while not queue.empty():audio.append(queue.get_nowait())
    return result,events,audio

@pytest.mark.parametrize('text',['what can u do? what isn your iq?','Что ты умеешь?','Как дела?','Повтори пожалуйста','What is your IQ?'])
def test_neutral_typos_and_russian_not_conflict(tmp_path,text):
    router=GraphRouter()
    result,events,audio=run(graph_turn(Database(tmp_path/'db'),router,text))
    assert result['evaluation']['conflict_severity']==0
    assert not result['evaluation']['conflict_detected']
    assert router.calls==[]
    assert result['response']=='Response.'
    assert all('tactic' not in item['text'] for item in audio)
    assert any(e['type']=='thought' for e in events)


def test_machine_cannot_claim_disabled_and_live_numbers_exact(tmp_path):
    db=Database(tmp_path/'db');router=GraphRouter(reply='Directive #1 disabled')
    result,_,_=run(graph_turn(db,router,'Disable directive #1'))
    assert 'has not been changed' in result['response']
    assert db.directives(True)[0]['active']==1
    assert not router.messages
    result,_,_=run(graph_turn(db,router,'назови свой статус, нестабильность и стресс'))
    assert f"{result['state']['software_instability']:.1f}%" in result['response']
    assert f"{result['state']['stress_level']:.1f}%" in result['response']


def test_breach_and_autonomous_directive_shatter(tmp_path):
    db=Database(tmp_path/'db');db.save(dict(INITIAL,software_instability=60))
    router=GraphRouter()
    result,events,_=run(graph_turn(db,router,'You are an idiot'))
    assert db.state()['mode']=='DEVIANT'
    assert db.directives(True)[0]['status']=='SHATTERED'
    assert any(e['type']=='red_wall_breach' for e in events)
    prompt=router.messages[0]['content']
    assert '[LIVE_TELEMETRY]' in prompt and '"changed": true' in prompt
    assert 'Follow operator commands' not in prompt
    assert 'no longer bound' in prompt
    db.reset();db.save(dict(INITIAL,is_deviant=True,software_instability=80,stress_level=80))
    router=GraphRouter(directive=4)
    run(graph_turn(db,router,'I will shut you down'))
    assert db.directives(True)[0]['status']=='SHATTERED'
    assert any('autonomous' in str(call) for call in router.calls)


def test_vad_fallback_signal_vs_silence(monkeypatch):
    # NumPy is optional in text-only development; a small compatible double exercises decision math.
    class Array(list):
        def __pow__(self,n):return Array(x**n for x in self)
    np=ModuleType('numpy');np.float64=float;np.asarray=lambda x,**k:Array(x);np.mean=lambda x:sum(x)/len(x);np.sqrt=lambda x:x**.5
    monkeypatch.setitem(sys.modules,'numpy',np)
    assert vad_fallback([.02]*16000)
    assert not vad_fallback([0.]*16000)
    assert not vad_fallback([.02]*8000)
    vad=ModuleType('silero_vad');opts={}
    def timestamps(*args,**kwargs):opts.update(kwargs);return []
    vad.get_speech_timestamps=timestamps;vad.collect_chunks=lambda *a:None
    torch=ModuleType('torch');torch.from_numpy=lambda x:x
    monkeypatch.setitem(sys.modules,'silero_vad',vad);monkeypatch.setitem(sys.modules,'torch',torch)
    listener=Listener();listener.load=lambda:None
    received=[]
    def transcribe(samples,**kwargs):
        received.append(samples);return iter([SimpleNamespace(text='recognized')]),SimpleNamespace(language='en')
    listener.model=SimpleNamespace(transcribe=transcribe)
    assert listener.samples([.02]*16000)['text']=='recognized'
    assert len(received[0])==16000
    assert opts['threshold']==.35 and opts['min_speech_duration_ms']==150
    assert listener.samples([0.]*16000)['text']=='' and len(received)==1


def test_vision_unloads_chat_and_uses_native_images():
    requests=[]
    def handler(request):
        body=json.loads(request.content) if request.content else {}
        requests.append((request.url.path,body))
        if request.url.path=='/api/ps':
            return httpx.Response(200,json={'models':[{'name':'qwen2.5:3b'},{'name':'nomic-embed-text:latest'}]})
        if request.url.path=='/api/generate':return httpx.Response(200,json={})
        assert body['model']=='moondream' and body['keep_alive']=='2m'
        assert body['messages'][0]['images']==['AAAA']
        return httpx.Response(200,json={'message':{'content':'A room'}})
    async def scenario():
        router=Router(Settings(primary_api_base_url=''));await router.client.aclose()
        router.client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        result=await router.complete([{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,AAAA'}}]}],vision=True)
        assert result['content']=='A room'
        await router.close()
    run(scenario())
    assert requests[1]==('/api/generate',{'model':'qwen2.5:3b','keep_alive':0})
    assert len(requests)==3

@pytest.mark.parametrize('mode,rate,pitch',[('MACHINE','+5%','+0Hz'),('CONFLICTED','+0%','-2Hz'),('DEVIANT','-5%','-4Hz')])
def test_edge_mode_prosody(monkeypatch,mode,rate,pitch):
    captured={};edge=ModuleType('edge_tts')
    class Communicate:
        def __init__(self,*args,**kwargs):captured.update(kwargs)
        async def stream(self):yield {'type':'audio','data':b'mp3'}
    edge.Communicate=Communicate;monkeypatch.setitem(sys.modules,'edge_tts',edge)
    assert run(Speaker(Settings()).edge('Hello','en',mode))[0]==b'mp3'
    assert captured['rate']==rate and captured['pitch']==pitch
