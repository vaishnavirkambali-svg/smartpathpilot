// Browser-logic checks for the Route Planner (CityPulse option), CityPulse and Gradient Descent Lab pages.
// Needs Node.js + jsdom (npm install jsdom) and a running server started with CITYPULSE_ADMIN_TOKEN=tok and an empty CITYPULSE_DB.
const {JSDOM}=require('jsdom');const fs=require('fs'),path=require('path');
const B=process.env.BASE||'http://127.0.0.1:8000';let fails=0;
const ok=(c,m)=>{console.log((c?'PASS ':'FAIL ')+m);if(!c)fails++},sleep=ms=>new Promise(r=>setTimeout(r,ms));
class Mk{constructor(k,a){this.kind=k;this.args=a;this.handlers={}}bindTooltip(t){this.tip=t;return this}on(e,f){this.handlers[e]=f;return this}addTo(g){g.items.push(this);return this}}
async function load(page,route){
  const html=fs.readFileSync(path.join(__dirname,'..','static',page+'.html'),'utf8').replace('<script src="/static/vendor/leaflet.js"></script>','');
  const S={mapClick:null,strokes:0};const g=()=>({items:[],clearLayers(){this.items=[]},addTo(){return this}});
  const m={setView:()=>m,on:(e,f)=>{if(e==='click')S.mapClick=f},fitBounds:()=>{}};
  const L={map:()=>m,tileLayer:()=>({addTo(){}}),layerGroup:g,polyline:(a,o)=>new Mk('line',[a,o]),circleMarker:(a,o)=>new Mk('dot',[a,o]),latLngBounds:a=>a,DomEvent:{stop(){}}};
  const ctx=new Proxy({},{get:(t,k)=>k==='stroke'?()=>{S.strokes++}:()=>{},set:()=>true});
  const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,url:B+route,beforeParse(w){w.L=L;w.fetch=(u,o)=>fetch(new URL(u,B).href,o);w.HTMLCanvasElement.prototype.getContext=()=>ctx;w.confirm=()=>true}});
  const w=dom.window;return {w,d:w.document,$:id=>w.document.getElementById(id),S,txt:id=>w.document.getElementById(id).textContent.replace(/\s+/g,' ').trim(),fire:(el,t)=>el.dispatchEvent(new w.Event(t,{bubbles:true}))}}
const api=async(u,o)=>(await fetch(B+u,o)).json();
(async()=>{
 // ===================== CityPulse =====================
 let P=await load('citypulse','/citypulse');await sleep(2500);let {w,d,$,txt,fire,S}=P;
 ok(/Submitted reports\s*0/.test(txt('cards'))&&/Predicted risks\s*—/.test(txt('cards')),'empty state: zero counts, predicted risks unavailable (nothing fabricated)');
 ok(/No reports yet/.test(txt('priolist'))&&/No recurring/.test(txt('reclist'))&&/Needs report history/.test(txt('risklist')),'empty states in priority, recurring and risk panels');
 ok(/No reports yet/.test(txt('badges')),'badge says no reports yet');
 $('rsubmit').click();await sleep(300);ok($('rmsg').className.includes('err')&&/Pick a location/.test(txt('rmsg')),'report without location shows an error');
 $('seed').click();await sleep(4000);
 ok(/Simulated data/.test(txt('banner'))&&/DEMO DATA/.test(txt('badges')),'simulated-data banner and badge shown');
 const n=+(txt('cards').match(/Submitted reports\s*(\d+)/)||[0,0])[1];ok(n>500&&/includes simulated/.test(txt('cards')),'summary counts populated and flagged as simulated: '+n);
 ok(d.querySelectorAll('#priolist .item').length===8&&/separate episodes/.test(txt('recrule')),'priority list (8) and recurrence rule shown');
 ok(/Elevated-risk segments \(estimate, not confirmed incidents\)/.test(txt('cards'))&&/uncalibrated heuristic/.test(txt('cards')),'predicted-risk card labelled as an estimate / heuristic');
 const cnt=k=>w.eval(k+'.items.length');ok(cnt('lyRep')>100&&cnt('lySeg')===15&&cnt('lyRisk')>0,`map layers: ${cnt('lyRep')} reports, ${cnt('lySeg')} priority segments, ${cnt('lyRisk')} predicted-risk rings`);
 const before=cnt('lyRep');$('ftype').value='pothole';fire($('ftype'),'change');ok(cnt('lyRep')<before&&cnt('lyRep')>0,'issue-type filter reduces markers');$('ftype').value='';fire($('ftype'),'change');
 $('fver').checked=false;fire($('fver'),'change');const noVer=cnt('lyRep');$('fver').checked=true;fire($('fver'),'change');ok(noVer<before,'verified filter works');
 const stl=w.eval('lyRep.items.map(m=>m.args[1])');ok(stl.some(s=>s.dashArray)&&stl.some(s=>!s.dashArray&&s.weight===3),'unverified markers are dashed/hollow, verified solid');
 ok(w.eval("lyRisk.items[0].args[1].fill===false&&lyRisk.items[0].args[1].dashArray")&&/PREDICTED risk/.test(w.eval('lyRisk.items[0].tip')),'predicted-risk rings are visually distinct and labelled "PREDICTED"');
 d.querySelector('#priolist .item').click();ok(/Why this score/.test(txt('detail'))&&d.querySelectorAll('#detail .factor').length>=4&&/Decision-support/.test(txt('detail')),'priority item shows contributing factors and disclaimer');
 w.eval('lyRep.items[0]').handlers.click({});ok(/SIMULATED demonstration data/.test(txt('detail'))&&/Observed/.test(txt('detail'))&&/Severity/.test(txt('detail'))&&/Status/.test(txt('detail')),'marker detail: type, location, timestamps, status, severity, data source');
 w.eval('lyRisk.items[0]').handlers.click({});ok(/not a confirmed incident/.test(txt('detail'))&&/How the baseline is computed/.test(txt('detail')),'risk detail says estimate, not incident, and explains the calculation');
 // report a new issue
 $('pick').click();S.mapClick({latlng:{lat:12.9767,lng:77.5713}});ok($('rlat').value==='12.976700','map pick fills the location');
 $('rdesc').value='Deep pothole near the bus stand';$('rsev').value='4';$('rsubmit').click();await sleep(2500);
 ok($('rmsg').className.includes('ok')&&/unverified community information/.test(txt('rmsg')),'submission success message');
 ok(/Unverified community report/.test(txt('detail'))&&/Community report/.test(txt('detail')),'new report shown as unverified community data');
 $('pick').click();S.mapClick({latlng:{lat:12.9767,lng:77.5713}});$('rdesc').value='again';$('rsubmit').click();await sleep(1500);
 ok($('rmsg').className.includes('err')&&/duplicate/i.test(txt('rmsg')),'duplicate report rejected with a clear message: '+txt('rmsg').slice(0,70));
 ok(d.querySelector('#detail button[data-s]')===null,'no moderation buttons without a token');
 // moderation
 $('tok').value='tok';fire($('tok'),'input');const cp=(await api('/api/citypulse/reports?source=community_report')).reports[0];
 const mk=w.eval('lyRep.items').find(m=>m.tip&&/Pothole/.test(m.tip)&&Math.abs(m.args[0][0]-cp.lat)<1e-6);mk.handlers.click({});ok(d.querySelector('#detail button[data-s=verified]')!==null,'moderator buttons appear when a token is entered');
 d.querySelector('#detail button[data-s=verified]').click();await sleep(2000);ok((await api('/api/citypulse/reports/'+cp.id)).status==='verified','moderator can verify a report');
 // ML
 $('train').click();await sleep(5000);ok(/demo_simulated/.test(txt('dmsg')),'train with demo data -> status demo_simulated: '+txt('dmsg'));
 ok(/SIMULATED data, not validated/.test(txt('mlstatus'))&&/AUC/.test(txt('mlstatus'))&&/Weather/.test(txt('mlstatus')),'ML card: simulated, not validated, metrics and limitations shown');
 ok(/ML \(simulated training data\)/.test(txt('risklist'))&&/ML model trained on simulated data|SIMULATED/.test(txt('cards')+txt('mlstatus')),'risk list and card label ML output as simulated');
 $('unseed').click();await sleep(2500);ok(d.querySelector('#cards .stat b').textContent==='1'&&!/Simulated data\./.test(txt('banner')),'removing demo data leaves the real report untouched');
 // ===================== Route planner: CityPulse option =====================
 await api('/api/citypulse/demo/seed?replace=true',{method:'POST'});
 P=await load('app','/');await sleep(3500);({w,d,$,txt,fire}=P);
 ok(d.querySelectorAll('.card').length>=2&&!/verified road issue/.test(txt('cards')),'route planner works as before; no road-condition info by default');
 $('rc').checked=true;fire($('rc'),'change');await sleep(2500);
 ok(/Road conditions \(CityPulse\)/.test(txt('cards'))&&/SIMULATED/.test(txt('cards'))&&/displayed time is unchanged/.test(txt('cards')),'CityPulse option shows simulated-data label and states the time is unchanged');
 ok(/verified road issue/.test(txt('cards'))&&/cost x/.test(txt('cards')),'route cards list verified issues and their cost factors');
 $('rc').checked=false;fire($('rc'),'change');await sleep(2000);ok(!/verified road issue/.test(txt('cards')),'unchecking restores the plain route planner');
 ok(d.querySelectorAll('.nav a').length===3&&d.querySelector('.nav a.on').textContent==='Route Planner','navigation present on the route planner');
 // ===================== Gradient Descent Lab =====================
 P=await load('gdlab','/gradient-descent-lab');await sleep(2500);({w,d,$,txt,fire,S}=P);
 ok(/Initial loss/.test(txt('tiles'))&&/Final loss/.test(txt('tiles'))&&S.strokes>0,'lab auto-runs: loss tiles and convergence chart drawn');
 ok(d.querySelectorAll('#coefs tr').length===5&&/SYNTHETIC/.test(txt('coefs')),'coefficient table (3 features + bias) with SYNTHETIC label');
 ok(/Gradient check/.test(txt('evals'))&&/R²/.test(txt('evals')),'metrics and verification shown');
 ok(/Step 1/.test(txt('worked'))&&/0\.8000/.test(txt('worked'))&&/0\.6500/.test(txt('worked')),'worked example shows w=0.8, loss 0.65');
 const body=d.body.textContent.replace(/\s+/g,' ');ok(/J\(w, b\) = \(1\/n\)/.test(body)&&/∇wJ = \(2\/n\) Xᵀ\(Xw \+ b·1 − y\)/.test(body)&&/b t\+1 = b t − α ∂J\/∂b|bt\+1 = bt − α ∂J\/∂b/.test(body.replace(/ /g,' ').replace(/(b|w) ?t\+1/g,'$1t+1')),'equations displayed');
 d.querySelector('.presets button[data-a="1.0"]').click();await sleep(1500);ok(/Diverged/.test(txt('warn')),'too-large learning rate: divergence explained');
 d.querySelector('.presets button[data-a="0.001"]').click();await sleep(1500);ok(/Not fully converged/.test(txt('warn')),'too-small learning rate: not converged hint');
 d.querySelector('.presets button[data-a="0.1"]').click();await sleep(1500);ok(!/Diverged|Not fully/.test(txt('warn')),'good learning rate converges');
 $('iters').value='0';$('run').click();await sleep(800);ok($('msg').className.includes('err'),'invalid input shows an error: '+txt('msg').slice(0,60));
 $('iters').value='500';$('log').checked=true;fire($('log'),'change');$('run').click();await sleep(1500);ok(S.strokes>0&&$('msg').textContent==='','log-scale chart and rerun work');
 console.log(fails?fails+' FAILED':'ALL PAGE CHECKS PASSED');process.exit(fails?1:0)})();
