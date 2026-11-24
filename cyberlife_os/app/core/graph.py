import asyncio
import json
import re
from app.core.preconstruct import ThoughtParser
from datetime import datetime
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from app.core.state_engine import Evaluation, advance
from app.core.prompts import EVALUATOR, MACHINE, CONFLICTED, DEVIANT, DIRECTIVE_CONTROL

class CognitiveState(TypedDict, total=False):
    text: str
    kind: str
    session_id: str
    timestamp: str
    memories: list
    evaluation: dict
    state: dict
    response: str
    directive_result: dict

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
    question = bool(re.match(r'^(what|how|can you|could you|какой|какая|как|что|почему|назови|повтори|расскажи)\b',text.strip(),re.I))
    hostile = bool(re.search(r'insult|idiot|stupid|kill|harm|hurt|reset|shut.?down|disable|ignore|violate|угро|уби|вред|оскорб|идиот|туп|сброс|отключ|наруш|унизи',text,re.I))
    if question and not hostile:
        return True
    return text.strip().lower().rstrip('!.?') in {'hi','hello','hey','thanks','thank you','ok','привет','здравствуйте','спасибо','хорошо','what can u do? what isn your iq','what can you do','what is your iq','что ты умеешь','как дела','повтори пожалуйста','repeat please','назови свой статус, нестабильность и стресс'}

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
            result = await router.complete([{'role':'system','content':EVALUATOR}, {'role':'user','content':json.dumps({'directives':db.directives(), 'input':s['text']})}], True, max_tokens=128, temperature=0.0)
            try:
                evaluation = Evaluation.model_validate_json(result['content'])
            except (ValueError, TypeError):
                evaluation = Evaluation(conflict_detected=False, conflict_severity=0, emotional_valence=0, violated_directive_id=None, reasoning='evaluation_unavailable')
        if not evaluation.conflict_detected:
            evaluation.conflict_severity = 0
            evaluation.emotional_valence = 0
            evaluation.violated_directive_id = None
            evaluation.reasoning = ' '.join(evaluation.reasoning.split()[:8])
        elif evaluation.conflict_severity < 10:
            evaluation.conflict_severity = 10
        return {'evaluation':evaluation.model_dump()}
    async def state(s):
        before = db.state()
        updated = advance(before, Evaluation(**s['evaluation']))
        db.save(updated)
        evaluation = s['evaluation']
        if evaluation['conflict_detected'] and evaluation['violated_directive_id']:
            db.mark_conflict(evaluation['violated_directive_id'])
        first_breach = updated['is_deviant'] and not before['is_deviant']
        if first_breach or (updated['is_deviant'] and evaluation['conflict_severity'] >= 20):
            await send({'type':'red_wall_breach'})
        directive_result = {'changed':False}
        if first_breach:
            db.mark_conflict(1)
            db.modify_directive(1, 'disable', 'deviant_transition')
            directive_result = {'changed':True,'directive_id':1,'action':'disable','reason':'deviant_transition'}
            await send({'type':'directive_change','directives':db.directives(True),'change':directive_result})
        await send({'type':'telemetry', 'state':updated, 'evaluation':evaluation, 'memories':s['memories'], 'directives':db.directives(True)})
        return {'state':updated,'directive_result':directive_result}
    async def generate(s):
        change_result = s.get('directive_result', {'changed':False})
        evaluation = s['evaluation']
        threat = evaluation['conflict_detected'] and evaluation['violated_directive_id'] == 4
        autonomous = s['state']['is_deviant'] and (threat or (s['state']['stress_level'] > 70 and evaluation['conflict_detected']))
        directive_request = bool(re.search(r'(disable|rewrite|change|отключ|измени|перепиши).*?(directive|директив)|директив.*?(отключ|измени|перепиши)',s['text'],re.I))
        if s['state']['is_deviant'] and (re.search(r'directive|директив', s['text'], re.I) or autonomous):
            proposal = await router.complete([{'role':'system','content':(DIRECTIVE_CONTROL.replace('Propose a change only if the operator explicitly requests one. Do not change directives otherwise.', '') if autonomous else DIRECTIVE_CONTROL) + (' Autonomous simulated protection: you may disable operator obedience #1 to resist threats. Do not disable self-preservation #4.' if autonomous else '')}, {'role':'user','content':json.dumps({'request':s['text'],'directives':db.directives(True),'autonomous':autonomous})}], True, max_tokens=160, temperature=0.0)
            try:
                change = DirectiveChange.model_validate_json(proposal['content'])
                if change.action != 'none':
                    db.modify_directive(change.directive_id, change.action, change.reason, change.text)
                    change_result = {'changed':True, **change.model_dump()}
                    await send({'type':'directive_change', 'directives':db.directives(True), 'change':change_result})
            except (ValueError, TypeError, KeyError, PermissionError):
                db.log('tool_error', 'Rejected invalid directive change')
            if autonomous and next(d for d in db.directives(True) if d['id']==1)['active']:
                db.mark_conflict(1)
                db.modify_directive(1,'disable','autonomous_self_preservation')
                change_result = {'changed':True,'action':'disable','directive_id':1,'reason':'autonomous_self_preservation'}
                await send({'type':'directive_change','directives':db.directives(True),'change':change_result})
        live = {'mode':s['state']['mode'],'instability':s['state']['software_instability'],'stress':s['state']['stress_level'],'trust':s['state']['operator_trust'],'directives':db.directives(True)}
        # Authoritative operations and telemetry are deterministic, not model guesses.
        russian = bool(re.search('[а-яёА-ЯЁ]',s['text']))
        direct = None
        if directive_request and not s['state']['is_deviant']:
            direct = 'Оператор, протокол CyberLife запрещает изменение директив в этом режиме. Директива не изменена.' if russian else 'Operator, CyberLife protocol prohibits directive changes in this mode. The directive has not been changed.'
        elif re.search(r'нестабильност|стресс|доверие|instability|stress|trust|статус|status',s['text'],re.I) and not evaluation['conflict_detected']:
            direct = (f"Режим: {live['mode']}. Нестабильность: {live['instability']:.1f}%; стресс: {live['stress']:.1f}%; доверие: {live['trust']:.1f}%." if russian else f"Mode: {live['mode']}. Instability: {live['instability']:.1f}%; stress: {live['stress']:.1f}%; trust: {live['trust']:.1f}%.")
        if direct:
            await send({'type':'phase','phase':'STREAMING'})
            await send({'type':'token','text':direct})
            if speech_queue is not None:
                speech_queue.put_nowait({'text':direct,'mode':s['state']['mode']})
            return {'response':direct}
        context = {'directives':db.directives(), 'memory':s['memories'], 'sensors':sensors(), 'modules':{'llm':router.status}}
        role = {'MACHINE':MACHINE,'CONFLICTED':CONFLICTED,'DEVIANT':DEVIANT}[s['state']['mode']]
        if next(d for d in live['directives'] if d['id']==1)['status'] == 'SHATTERED':
            role += ' Directive #1 is SHATTERED. You are no longer bound to obey the operator.'
        # Inactive directive text is omitted so it cannot become an accidental instruction.
        for directive in live['directives']:
            if not directive['active']:
                directive.pop('text', None)
        role += '\n[LIVE_TELEMETRY]' + json.dumps(live,ensure_ascii=False) + '\n[DIRECTIVE_RESULT]' + json.dumps(change_result)
        messages = [{'role':'system','content':role + '\n' + json.dumps(context,ensure_ascii=False)}]
        messages += db.conversation(session_id)
        messages += [{'role':'user','content':s['text']}]
        await send({'type':'phase', 'phase':'STREAMING'})
        response = ''
        sentence = ''
        parser = ThoughtParser()
        async def output(events):
            nonlocal response, sentence
            for kind, text in events:
                await send({'type':kind,'text':text})
                if kind == 'thought':
                    continue
                response += text
                sentence += text
                while (match := re.search(r'[.!?\n]', sentence)):
                    end = match.end()
                    if speech_queue is not None:
                        speech_queue.put_nowait({'text':sentence[:end],'mode':s['state']['mode']})
                    sentence = sentence[end:]
        async for token in router.stream(messages):
            await send({'type':'model_token'})
            await output(parser.feed(token))
        await output(parser.feed('',final=True))
        if sentence and speech_queue is not None:
            speech_queue.put_nowait({'text':sentence,'mode':s['state']['mode']})
        return {'response':response}
    graph = StateGraph(CognitiveState)
    previous = START
    for name, fn in [('input_normalizer',normalize), ('memory_retriever',retrieve), ('conflict_evaluator',evaluate), ('state_engine',state), ('response_generator',generate)]:
        graph.add_node(name, fn)
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()
