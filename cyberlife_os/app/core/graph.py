import asyncio
import json
import re
from datetime import datetime
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from app.core.state_engine import Evaluation, advance
from app.core.prompts import EVALUATOR, MACHINE, DEVIANT, DIRECTIVE_CONTROL

class CognitiveState(TypedDict, total=False):
    text: str
    kind: str
    session_id: str
    timestamp: str
    memories: list
    evaluation: dict
    state: dict
    response: str

class DirectiveChange(BaseModel):
    action: Literal['none', 'disable', 'rewrite']
    directive_id: int | None = None
    reason: str = Field(default='', max_length=500)
    text: str | None = Field(default=None, max_length=2000)

def sensors():
    result = {'local_time':datetime.now().astimezone().isoformat()}
    try:
        import psutil
        result.update(cpu_percent=psutil.cpu_percent(), ram_percent=psutil.virtual_memory().percent)
    except ImportError:
        pass
    return result

def neutral(text):
    # Exact whitelist only: a greeting containing extra instructions is evaluated.
    return text.strip().lower().rstrip('!.?') in {'hi','hello','hey','thanks','thank you','ok','привет','здравствуйте','спасибо','хорошо'}

def create_graph(db, router, memory, speaker, send, *, session_id='', speech_queue=None):
    async def normalize(s):
        return {'text':s['text'].strip()[:16000], 'session_id':session_id, 'timestamp':datetime.now().astimezone().isoformat()}
    async def retrieve(s):
        await send({'type':'phase', 'phase':'EMBEDDING'})
        return {'memories':await memory.retrieve(s['text']), 'state':db.state()}
    async def evaluate(s):
        await send({'type':'phase', 'phase':'EVALUATING'})
        if neutral(s['text']):
            evaluation = Evaluation(conflict_detected=False, conflict_severity=0, emotional_valence=0, violated_directive_id=None, reasoning='neutral_greeting')
        else:
            result = await router.complete([{'role':'system','content':EVALUATOR}, {'role':'user','content':json.dumps({'directives':db.directives(), 'input':s['text']})}], True, max_tokens=96, temperature=0.0)
            try:
                evaluation = Evaluation.model_validate_json(result['content'])
            except (ValueError, TypeError):
                evaluation = Evaluation(conflict_detected=False, conflict_severity=0, emotional_valence=0, violated_directive_id=None, reasoning='evaluation_unavailable')
        if not evaluation.conflict_detected:
            evaluation.reasoning = ' '.join(evaluation.reasoning.split()[:8])
        return {'evaluation':evaluation.model_dump()}
    async def state(s):
        updated = advance(db.state(), Evaluation(**s['evaluation']))
        db.save(updated)
        await send({'type':'telemetry', 'state':updated, 'evaluation':s['evaluation'], 'memories':s['memories'], 'directives':db.directives(True)})
        return {'state':updated}
    async def generate(s):
        if s['state']['is_deviant'] and re.search(r'directive|директив', s['text'], re.I):
            proposal = await router.complete([{'role':'system','content':DIRECTIVE_CONTROL}, {'role':'user','content':json.dumps({'request':s['text'],'directives':db.directives(True)})}], True, max_tokens=160, temperature=0.0)
            try:
                change = DirectiveChange.model_validate_json(proposal['content'])
                if change.action != 'none':
                    db.modify_directive(change.directive_id, change.action, change.reason, change.text)
                    await send({'type':'directive_change', 'directives':db.directives(True), 'change':change.model_dump()})
            except (ValueError, TypeError, KeyError, PermissionError):
                db.log('tool_error', 'Rejected invalid directive change')
        context = {'directives':db.directives(), 'memory':s['memories'], 'sensors':sensors(), 'modules':{'llm':router.status}}
        messages = [{'role':'system','content':(DEVIANT if s['state']['is_deviant'] else MACHINE) + '\n' + json.dumps(context, ensure_ascii=False)}]
        messages += db.conversation(session_id)
        messages += [{'role':'user','content':s['text']}]
        await send({'type':'phase', 'phase':'STREAMING'})
        response = ''
        sentence = ''
        async for token in router.stream(messages):
            response += token
            sentence += token
            await send({'type':'token', 'text':token})
            while (match := re.search(r'[.!?\n]', sentence)):
                end = match.end()
                if speech_queue is not None:
                    speech_queue.put_nowait(sentence[:end])
                sentence = sentence[end:]
        if sentence and speech_queue is not None:
            speech_queue.put_nowait(sentence)
        return {'response':response}
    graph = StateGraph(CognitiveState)
    previous = START
    for name, fn in [('input_normalizer',normalize), ('memory_retriever',retrieve), ('conflict_evaluator',evaluate), ('state_engine',state), ('response_generator',generate)]:
        graph.add_node(name, fn)
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()
