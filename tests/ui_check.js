const {JSDOM}=require('jsdom');const fs=require('fs'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'..','static','app.html'),'utf8').replace('<script src="/static/vendor/leaflet.js"></script>','');
let clicks,fits=0,lines=0;
const m={setView:()=>m,on:(e,f)=>{if(e==='click')clicks=f},fitBounds:()=>{fits++}};
const g={clearLayers(){},addTo(){return g}};
const L={map:()=>m,tileLayer:()=>({addTo(){}}),layerGroup:()=>g,polyline:()=>{lines++;return{addTo(){}}},circleMarker:()=>({bindTooltip:()=>({addTo(){}})}),latLngBounds:a=>a};
const B=process.env.BASE||'http://127.0.0.1:8000';
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,url:B+'/',beforeParse(w){w.L=L;w.fetch=(u,o)=>fetch(new URL(u,B).href,o)}});
const w=dom.window,d=w.document,sleep=ms=>new Promise(r=>setTimeout(r,ms)),$=id=>d.getElementById(id);
const txt=id=>$(id).textContent.replace(/\s+/g,' ').trim();let fails=0;const ok=(c,m)=>{console.log((c?'PASS ':'FAIL ')+m);if(!c)fails++};
const fire=(el,t)=>el.dispatchEvent(new w.Event(t,{bubbles:true}));
(async()=>{
 await sleep(3000);
 ok($('srcq').value==='Majestic'&&$('dstq').value==='Whitefield','defaults loaded: '+$('srcq').value+' -> '+$('dstq').value);
 ok(d.querySelectorAll('.card').length>=2&&lines>0&&fits>0,'cards + map lines drawn');
 ok(txt('badges').includes('SIMULATED')&&txt('badges').includes('DEMO'),'provenance badges');
 ok(d.querySelectorAll('#hot .hr').length===5&&/simulated/.test(txt('hoth')),'hotspots listed: '+txt('hoth'));
 // search + select destination
 $('dstq').value='cubb';fire($('dstq'),'input');await sleep(2500);
 ok(txt('dsts').includes('Cubbon Park'),'search suggests Cubbon Park');
 const it=[...$('dsts').querySelectorAll('div[data-i]')].find(x=>x.textContent.includes('Cubbon Park'));
 it.onmousedown({preventDefault(){}});await sleep(1500);
 ok($('dstq').value==='Cubbon Park'&&d.querySelectorAll('.card').length>=1,'selecting a suggestion routes to Cubbon Park');
 // avoid poor roads Majestic->Whitefield
 $('dstq').value='';$('dstq').onfocus&&0;
 const wf=await (await fetch(B+'/api/osm/search?q=whitefield&online=false')).json();
 // set via search UI
 $('dstq').value='whitef';fire($('dstq'),'input');await sleep(2500);
 [...$('dsts').querySelectorAll('div[data-i]')][0].onmousedown({preventDefault(){}});await sleep(1500);
 ok($('dstq').value==='Whitefield','destination set to Whitefield via search');
 const before=txt('steps');
 $('ap').checked=true;fire($('ap'),'change');await sleep(1500);
 ok(txt('steps')!==before&&!txt('steps').includes('Whitefield Main Rd'),'avoid poor roads changes the route');
 $('aw').checked=true;fire($('aw'),'change');await sleep(1500);
 ok($('msg').className==='err'&&/filters/i.test(txt('msg')),'impossible filters show clear error: '+txt('msg'));
 $('ap').checked=false;$('aw').checked=false;fire($('ap'),'change');await sleep(1500);
 ok($('msg').className!=='err'&&d.querySelectorAll('.card').length>=2,'recovers after filters removed');
 // (engine checks below)
 $('eng').value='ml';fire($('eng'),'change');await sleep(1500);ok(txt('badges').includes('ML PREDICTION'),'ML badge');
 $('eng').value='live';fire($('eng'),'change');await sleep(1500);
 ok(!txt('badges').includes('LIVE')&&txt('badges').includes('(fallback)')&&$('h').disabled,'live without key: labelled fallback, never LIVE');
 $('eng').value='profile';fire($('eng'),'change');await sleep(1200);
 // map clicks, swap, best time
 clicks({latlng:{lat:12.935,lng:77.6245}});clicks({latlng:{lat:13.036,lng:77.597}});await sleep(1500);
 ok(/Map point/.test($('srcq').value)&&d.querySelectorAll('.card').length>=2,'map clicks set points');
 $('swap').click();await sleep(1500);ok(/Map point/.test($('dstq').value),'swap works');
 $('bt').click();await sleep(2500);ok(d.querySelectorAll('.bars div').length===24,'best-time chart');

 // ===== multi-stop panel =====
 const opts=[...$('mtp').options].map(o=>o.value+(o.disabled?':disabled':''));
 ok(opts.join()==='free_flow,typical,peak,live:disabled'&&/NOT LIVE/.test(txt('mtpnote')),'traffic profiles listed, live disabled when not configured: '+opts.join());
 const addStop=async q=>{$('stopq').value=q;fire($('stopq'),'input');await sleep(2200);[...$('stopbox').querySelectorAll('div[data-i]')][0].onmousedown({preventDefault(){}});await sleep(200)};
 $('srcq').value='majestic';fire($('srcq'),'input');await sleep(2200);[...$('srcs').querySelectorAll('div[data-i]')][0].onmousedown({preventDefault(){}});await sleep(1200);
 $('dstq').value='whitef';fire($('dstq'),'input');await sleep(2200);[...$('dsts').querySelectorAll('div[data-i]')][0].onmousedown({preventDefault(){}});await sleep(1200);
 for(const q of ['silk board','hebbal','jayanagar'])await addStop(q);
 ok(d.querySelectorAll('#stoplist .stoprow').length===3&&/Silk Board/.test(txt('stoplist')),'3 stops added: '+txt('stoplist'));
 d.querySelector('#stoplist button[data-a=up][data-i="2"]').click();
 ok(/1\. Silk Board\s*2\. Jayanagar\s*3\. Hebbal/.test(txt('stoplist').replace(/[\u2191\u2193\u2715]/g,'')),'reorder up works: '+txt('stoplist').replace(/[\u2191\u2193\u2715]/g,''));
 d.querySelector('#stoplist button[data-a=down][data-i="1"]').click();
 d.querySelector('#stoplist button[data-a=del][data-i="1"]').click();
 ok(d.querySelectorAll('#stoplist .stoprow').length===2,'remove works');
 await addStop('hebbal');
 w.eval("st.stops.push({name:'<img src=x onerror=1>',lat:12.9,lon:77.6});renderStops()");
 ok(d.querySelector('#stoplist img')===null&&/<img/.test($('stoplist').textContent),'stop names are escaped (no HTML injection)');
 w.eval("st.stops.pop();renderStops()");
 // compare
 $('mmode').value='compare';$('mgo').click();await sleep(3500);
 ok(d.querySelectorAll('#mout .card').length===2&&/Optimized order/.test(txt('mout'))&&/Strict order/.test(txt('mout')),'compare shows both results');
 ok(/saves|already optimal/.test(txt('mout'))&&/NOT LIVE/.test(txt('mout')),'comparison + NOT LIVE label: '+(txt('mout').match(/Optimized order saves[^.]*\./)||[''])[0]);
 ok(/Quality penalty/.test(txt('mout'))&&/Exact order search/.test(txt('mout')),'quality + optimization details shown');
 const lines0=lines;
 d.querySelectorAll('#mout .card')[1].click();await sleep(300);ok(lines>lines0&&d.querySelectorAll('#mout .card.sel')[0].dataset.k==='strict','selecting a result redraws the map');
 // strict
 $('mmode').value='strict';$('mgo').click();await sleep(2500);
 const strictOrder=txt('mout').match(/Majestic →[^·]*?Whitefield/);ok(d.querySelectorAll('#mout .card').length===1&&strictOrder&&strictOrder[0].replace(/\s+/g,' ').trim()==='Majestic → Silk Board → Jayanagar → Hebbal → Whitefield','strict keeps entered order: '+(strictOrder||[''])[0]);
 // objective + road preference
 $('mmode').value='optimized';$('mobj').value='shortest';$('mrq').value='better_roads';$('mgo').click();await sleep(2500);
 ok(d.querySelectorAll('#mout .card').length===1&&$('mmsg').className!=='err','shortest + better roads works');
 // weather: server has no internet here -> must say unavailable, still show route
 $('mobj').value='fastest';$('mrq').value='none';$('mwx').checked=true;$('mgo').click();await sleep(4500);
 ok(d.querySelectorAll('#mout .card').length===1&&(/Weather unavailable|Weather \(open-meteo\)/.test(txt('mout'))),'weather: '+(txt('mout').match(/Weather[^.]*\./)||['none'])[0].slice(0,90));
 $('mwx').checked=false;
 // errors
 w.eval("st.stops=[];renderStops();st.dst=st.src;");$('mgo').click();await sleep(1500);
 ok($('mmsg').className==='err'&&/same junction/.test(txt('mmsg')),'same source/destination error shown: '+txt('mmsg'));
 console.log(fails?fails+' FAILED':'ALL UI CHECKS PASSED');process.exit(fails?1:0)})();
