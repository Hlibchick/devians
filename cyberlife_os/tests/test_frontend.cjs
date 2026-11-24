// Protocol/UI checks with deterministic DOM and audio doubles; no npm dependencies.
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const elements=new Map();
function element(){return {textContent:'',value:'',style:{},options:[],children:[],append(...x){this.children.push(...x);},replaceChildren(){this.children=[];},setAttribute(){}};}
const players=[];
class Audio {constructor(){players.push(this);} play(){return Promise.resolve();} pause(){this.paused=true;}}
class WebSocket {static OPEN=1;constructor(){this.readyState=1;this.sent=[];}send(x){this.sent.push(JSON.parse(x));}}
const context=vm.createContext({document:{getElementById:id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);},createElement:element},Audio,WebSocket,location:{protocol:'http:',host:'localhost'},crypto:{randomUUID:()=> 'next'},setTimeout:()=>0,clearTimeout(){},cancelAnimationFrame(){},console});
vm.runInContext(fs.readFileSync('app/static/app.js','utf8'),context);
vm.runInContext(`
turnId='one';
ws.onmessage({data:JSON.stringify({type:'audio_chunk',turn_id:'one',mime:'audio/wav',data_b64:'AA==',seq_id:0})});
ws.onmessage({data:JSON.stringify({type:'audio_chunk',turn_id:'one',mime:'audio/wav',data_b64:'AA==',seq_id:1})});
`,context);
assert.equal(players.length,1,'Second sentence must wait');
players[0].onended();assert.equal(players.length,2);
vm.runInContext("startTurn('new message')",context);
assert.equal(players[1].paused,true,'Barge-in stops audio');
vm.runInContext("ws.onmessage({data:JSON.stringify({type:'audio_chunk',turn_id:'one',mime:'audio/wav',data_b64:'AA=='})})",context);
assert.equal(players.length,2,'Old turn audio is ignored');
vm.runInContext("$('mute').onclick(); ws.onmessage({data:JSON.stringify({type:'audio_chunk',turn_id:'next',mime:'audio/wav',data_b64:'AA=='})})",context);
assert.equal(players.length,2,'Mute suppresses playback');
console.log('Frontend: ordered audio, barge-in, stale turn rejection, mute passed');
// Default selection follows backend configuration, never the first heavy model.
(async()=>{
  context.fetch=async()=>({ok:true,json:async()=>({models:[{id:'heavy:7b',provider:'local',vram_warning:true},{id:'qwen2.5:3b',provider:'local'}],preferred_model:'qwen2.5:3b',active:{provider:'local',model:'qwen2.5:3b'}})});
  await vm.runInContext('loadModels()',context);
  assert.equal(elements.get('model').value,'qwen2.5:3b');
  assert.equal(elements.get('model').children[0].textContent,'heavy:7b [VRAM > 4GB]');
  vm.runInContext("ws.onmessage({data:JSON.stringify({type:'thought',turn_id:'next',text:'fictional tactic'})})",context);
  assert.equal(elements.get('thought').textContent,'fictional tactic');
  console.log('Frontend v3: configured default, VRAM warning and PRECONSTRUCT passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
