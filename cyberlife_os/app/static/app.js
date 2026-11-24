const $ = id => document.getElementById(id);
let ws, turnId = null, current = null, audio = null, muted = false, modelList = [];
let audioQueue = [], playing = false, recorder = null, mediaStream = null, micContext = null;
let meterFrame = null, recordTimer = null, sttController = null, captureVersion = 0;
function notice(text) { $('notice').textContent = text; }
function send(value) {
  if (!ws || ws.readyState !== WebSocket.OPEN) { notice('Нет соединения'); return false; }
  ws.send(JSON.stringify(value)); return true;
}
function clearAudio() {
  audioQueue = [];
  if (audio) { audio.onended = audio.onerror = null; audio.pause(); audio.src = ''; audio = null; }
  playing = false;
}
function stopTurn() {
  if (recorder?.state === 'recording') recorder.stop();
  clearAudio(); send({type:'stop'}); turnId = null; $('phase').textContent = 'IDLE';
  if (sttController) { sttController.abort(); sttController = null; }
  captureVersion++;
}
function playNext() {
  if (playing || muted || !audioQueue.length) return;
  const chunk = audioQueue.shift();
  if (chunk.turn_id !== turnId) { playNext(); return; }
  const item = new Audio(`data:${chunk.mime};base64,${chunk.data_b64}`);
  audio = item; playing = true;
  const next = () => { if (audio !== item) return; playing = false; audio = null; playNext(); };
  item.onended = next; item.onerror = next;
  item.play().catch(() => { notice('Аудио заблокировано браузером. Разрешите воспроизведение.'); next(); });
}
function message(role, text='') {
  const box = document.createElement('article'); box.className = 'message ' + role;
  const label = document.createElement('div'); label.className = 'label'; label.textContent = role === 'operator' ? 'OPERATOR' : 'CYBERLIFE OS';
  const body = document.createElement('div'); body.className = 'body'; body.textContent = text;
  const timing = document.createElement('div'); timing.className = 'timing'; box.append(label, body, timing); $('chat').append(box);
  return {body, timing};
}
function startTurn(text, type='text') {
  if (!ws || ws.readyState !== WebSocket.OPEN) { notice('Нет соединения'); return; }
  stopTurn(); turnId = crypto.randomUUID();
  message('operator', text || (type === 'vision_event' ? '[Camera]' : '[Server microphone]'));
  current = message('assistant'); $('phase').textContent = 'QUEUED';
  send({type, text, turn_id:turnId});
}
function renderDirectives(items) {
  $('directives').replaceChildren();
  for (const d of items) {
    const item = document.createElement('div'); item.className = 'directive' + (!d.active ? ' disabled' : d.rewritten ? ' rewritten' : '');
    item.textContent = `#${d.id} · ${!d.active ? 'OFF' : d.rewritten ? 'REWRITTEN' : 'ACTIVE'} — ${d.text}`;
    $('directives').append(item);
  }
}
function telemetry(d) {
  if (d.state) {
    const s = d.state; $('mode').textContent = s.is_deviant ? 'DEVIANT' : 'MACHINE';
    const color = {BLUE:'#39aaff',YELLOW:'#ffcf40',RED:'#ff4040'}[s.led_status];
    $('led').style.background = color; $('led').style.boxShadow = `0 0 18px ${color}`;
    $('metrics').replaceChildren();
    for (const k of ['software_instability','stress_level','operator_trust']) {
      const label = document.createElement('label'); label.className = 'metric'; label.textContent = `${k}: ${s[k].toFixed(1)}`;
      const p = document.createElement('progress'); p.max = 100; p.value = s[k]; label.append(p); $('metrics').append(label);
    }
  }
  if (d.directives) renderDirectives(d.directives);
  if (d.evaluation) { const e=d.evaluation; $('evaluation').textContent=`${e.conflict_detected ? 'CONFLICT' : 'NORMAL'} · severity ${e.conflict_severity} · valence ${e.emotional_valence}\n${e.reasoning}`; }
  if (d.memories) {
    $('memories').replaceChildren();
    for (const m of d.memories) { const item=document.createElement('div'); item.className='card'; item.textContent=m.text; $('memories').append(item); }
    if (!d.memories.length) $('memories').textContent='Нет релевантных фактов';
  }
  $('log').textContent = JSON.stringify(d, null, 2);
}
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { notice('Подключено'); loadModels(); };
  ws.onmessage = event => {
    const d=JSON.parse(event.data);
    if (d.turn_id && d.turn_id !== turnId) return;
    if (d.type === 'ready') $('tts').value=d.tts_engine;
    if (d.type === 'telemetry' || d.type === 'directive_change') telemetry(d);
    if (d.type === 'phase') $('phase').textContent=d.phase;
    if (d.type === 'token' && current) {
      current.body.textContent += d.text;
      if (d.ttft_ms != null) current.timing.textContent=`TTFT ${d.ttft_ms} ms`;
      $('chat').scrollTop=$('chat').scrollHeight;
    }
    if (d.type === 'done') { $('phase').textContent='IDLE'; if(current) current.timing.textContent=`TTFT ${d.ttft_ms ?? '—'} ms · Total ${d.total_ms} ms · ${d.provider}`; }
    if (d.type === 'audio_chunk' && !muted) { audioQueue.push(d); playNext(); }
    if (d.type === 'error' || d.type === 'warning') { notice(d.message); if(d.type==='error') $('phase').textContent='ERROR'; }
    if (d.type === 'transcript') notice(`STT (${d.language}): ${d.text}`);
  };
  ws.onclose = () => { clearAudio(); turnId=null; notice('Соединение потеряно; переподключение…'); setTimeout(connect, 2000); };
}
function renderModels() {
  $('model').replaceChildren();
  for (const m of modelList.filter(m => m.provider === $('provider').value)) { const o=document.createElement('option'); o.value=m.id; o.textContent=m.id; $('model').append(o); }
  $('model').disabled = !$('model').options.length;
}
async function loadModels() {
  try { const r=await fetch('/api/models'); if(!r.ok) throw Error('Модели недоступны'); const d=await r.json(); modelList=d.models; $('provider').value=d.active.provider; renderModels(); if(d.active.model) $('model').value=d.active.model; if(!modelList.length) notice('Ollama недоступна или модели не установлены — диагностический режим.'); }
  catch(e) { notice(e.message); }
}
function chooseModel() { if($('model').value) { stopTurn(); send({type:'provider',provider:$('provider').value,model:$('model').value}); } }
async function cleanupMic() {
  clearTimeout(recordTimer); cancelAnimationFrame(meterFrame);
  if(mediaStream) mediaStream.getTracks().forEach(t=>t.stop()); mediaStream=null;
  if(micContext) await micContext.close(); micContext=null;
  $('mic').textContent='🎙 Запись'; $('mic').setAttribute('aria-pressed','false'); $('level').value=0;
}
async function toggleMic() {
  if(recorder && recorder.state==='recording') { recorder.stop(); return; }
  stopTurn(); const version=captureVersion;
  try {
    if(!navigator.mediaDevices || !window.MediaRecorder) throw Error('Нужен HTTPS/localhost и разрешение микрофона. Доступен текст или опциональный микрофон сервера.');
    mediaStream=await navigator.mediaDevices.getUserMedia({audio:true});
    if(version!==captureVersion) { await cleanupMic(); return; }
    const chunks=[]; recorder=new MediaRecorder(mediaStream);
    recorder.ondataavailable=e=>{if(e.data.size) chunks.push(e.data);};
    recorder.onstop=async()=>{
      const type=recorder.mimeType; await cleanupMic();
      if(version!==captureVersion) return;
      const controller=new AbortController(); sttController=controller;
      try {
        notice('Распознавание…');
        const r=await fetch('/api/stt',{method:'POST',headers:{'Content-Type':type},body:new Blob(chunks,{type}),signal:controller.signal});
        const d=await r.json(); if(!r.ok) throw Error(d.detail || 'STT error');
        if(version!==captureVersion) return;
        sttController=null; if(d.text) startTurn(d.text); else notice('Речь не обнаружена');
      } catch(e) { if(e.name!=='AbortError') notice(e.message); }
    };
    recorder.start(250); $('mic').textContent='■ Завершить запись'; $('mic').setAttribute('aria-pressed','true');
    micContext=new AudioContext(); const analyser=micContext.createAnalyser(); analyser.fftSize=256;
    micContext.createMediaStreamSource(mediaStream).connect(analyser); const samples=new Uint8Array(analyser.fftSize);
    const meter=()=>{analyser.getByteTimeDomainData(samples); $('level').value=Math.min(1,Math.sqrt(samples.reduce((s,x)=>s+((x-128)/128)**2,0)/samples.length)*5); meterFrame=requestAnimationFrame(meter);}; meter();
    recordTimer=setTimeout(()=>{if(recorder.state==='recording') recorder.stop();},120000);
  } catch(e) { await cleanupMic(); notice(e.message); }
}
$('form').onsubmit=e=>{e.preventDefault(); const text=$('input').value.trim(); if(text) {startTurn(text);$('input').value='';}};
$('stop').onclick=()=>{stopTurn(); if(recorder?.state==='recording') recorder.stop();};
$('stopAudio').onclick=stopTurn;
$('mute').onclick=()=>{muted=!muted;clearAudio();$('mute').textContent=muted?'Unmute':'Mute';$('mute').setAttribute('aria-pressed',String(muted));};
$('tts').onchange=()=>{stopTurn();send({type:'tts',engine:$('tts').value});};
$('mic').onclick=toggleMic;
$('serverMic').onclick=()=>startTurn('', 'voice');
$('camera').onclick=()=>startTurn('', 'vision_event');
$('provider').onchange=()=>{renderModels();chooseModel();};$('model').onchange=chooseModel;$('refreshModels').onclick=loadModels;
$('reset').onclick=()=>{if(confirm('Удалить память и восстановить директивы?')){stopTurn();send({type:'reset'});$('chat').replaceChildren();$('evaluation').textContent='Состояние сброшено';}};
connect();
