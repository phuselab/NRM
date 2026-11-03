(() => {
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = {a:css('--a'), b:css('--b'), c:css('--c'), d:css('--d'), thr:css('--thr'), ink:css('--ink'), muted:css('--muted'), rule:css('--rule')};
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// seeded RNG + gaussian
function rng(seed){ let s = seed>>>0 || 1; return () => { s ^= s<<13; s>>>=0; s ^= s>>17; s ^= s<<5; s>>>=0; return (s>>>0)/4294967296; }; }
function gauss(r){ let u=0,v=0; while(u===0)u=r(); while(v===0)v=r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

// Resize a canvas for the current device pixel ratio and return a context whose
// coordinates are CSS pixels. The intended CSS height is latched in `data-h` on
// first use: `cv.height` is the backing-store size, so reading it back on a
// later call would compound the dpr scaling and grow the canvas without bound.
function setup(cv, h){
  const dpr = window.devicePixelRatio || 1;
  if(cv.dataset.h === undefined || h) cv.dataset.h = h || cv.getAttribute('height') || cv.clientHeight || 200;
  const hh = +cv.dataset.h;
  const w = cv.clientWidth || cv.parentElement.clientWidth || 420;
  cv.style.height = hh + 'px';
  cv.width = Math.round(w*dpr); cv.height = Math.round(hh*dpr);
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w, h:hh};
}

/* ---------------- HERO ---------------- */
const heroCv = document.getElementById('hero');
const heroOut = document.getElementById('heroReadout');
const LABELS = ['cat','dog','car','boat'];
const HCOL = [COL.a, COL.b, COL.c, COL.d];
// Frame and replay handles are kept apart: one needs cancelAnimationFrame, the
// other clearTimeout, and cancelling with the wrong one leaves a second loop
// running on top of the first.
let heroSeed = 7, heroFrame = null, heroReplay = null;
function heroStop(){
  if(heroFrame !== null){ cancelAnimationFrame(heroFrame); heroFrame = null; }
  if(heroReplay !== null){ clearTimeout(heroReplay); heroReplay = null; }
}
function heroTrial(){
  const r = rng(heroSeed++*9973);
  const target = Math.floor(r()*4);
  const strength = 0.6 + r()*2.2;
  const drifts = LABELS.map((_,i)=> 0.55 + (i===target?strength:0) + 0.35*gauss(r));
  const theta = 1.1 + 0.5*r();          // stimulus-dependent bound
  const sig = 0.75, dt = 0.004, T = 2.0, n = Math.round(T/dt);
  const X = LABELS.map(()=>new Float32Array(n+1));
  let hit = -1, win = -1;
  for(let j=1;j<=n;j++){
    for(let i=0;i<4;i++){ X[i][j] = X[i][j-1] + drifts[i]*dt + sig*Math.sqrt(dt)*gauss(r); }
    if(hit<0){ for(let i=0;i<4;i++){ if(X[i][j]>=theta){ hit=j; } } if(hit>0){ let m=-1e9; for(let i=0;i<4;i++){ if(X[i][j]>m){m=X[i][j];win=i;} } } }
  }
  if(hit<0){ hit=n; let m=-1e9; for(let i=0;i<4;i++) if(X[i][n]>m){m=X[i][n];win=i;} }
  // How legible the stimulus is: the same quantity that sets the target's drift.
  const clarity = (strength - 0.6) / 2.2;
  return {X, n, dt, theta, hit, win, target, T, clarity, seed: heroSeed};
}
// The stimulus panel: the target's photo, degraded in proportion to how weak
// its drift is — blur and desaturation on the image, a grain layer on top.
const heroStim = document.getElementById('heroStim');
const heroNoise = document.getElementById('heroNoise');
const heroCap = document.getElementById('heroStimCap');
const heroClar = document.getElementById('heroClarity');
const heroClarV = document.getElementById('heroClarityV');
function heroShowStimulus(tr){
  const deg = 1 - tr.clarity;
  if(heroStim){
    heroStim.src = 'assets/stimuli/' + LABELS[tr.target] + '.jpg';
    heroStim.style.filter = `blur(${(4.5*deg).toFixed(1)}px) grayscale(${(0.6*deg).toFixed(2)}) contrast(${(1-0.3*deg).toFixed(2)})`;
  }
  if(heroNoise){
    // Sized to its frame (a square set by CSS), not latched like the charts.
    const dpr = window.devicePixelRatio || 1;
    const w = heroNoise.clientWidth || 180, h = heroNoise.clientHeight || w;
    heroNoise.width = Math.round(w*dpr); heroNoise.height = Math.round(h*dpr);
    const ctx = heroNoise.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,w,h);
    const r = rng(tr.seed*7919), cells = 110, cw = w/cells, ch = h/cells;
    for(let y=0;y<cells;y++) for(let x=0;x<cells;x++){
      const v = Math.floor(r()*255);
      ctx.fillStyle = `rgba(${v},${v},${v},${(0.5*deg*deg).toFixed(3)})`;
      ctx.fillRect(x*cw, y*ch, cw+0.5, ch+0.5);
    }
  }
  // The label takes the colour of its accumulator, tying the image to its trace.
  const pct = Math.round(100*tr.clarity);
  if(heroCap){ heroCap.textContent = LABELS[tr.target]; heroCap.style.color = HCOL[tr.target]; }
  if(heroClarV) heroClarV.textContent = pct + '%';
  if(heroClar){ heroClar.style.width = Math.max(4, pct) + '%'; heroClar.style.backgroundColor = HCOL[tr.target]; }
}
function heroDraw(tr, upto){
  const {ctx,w,h} = setup(heroCv, window.innerWidth<600?240:320);
  // Narrow canvases (phones, beside the stimulus card) get a tighter margin
  // and one tick per second so the axis labels never collide.
  const narrow = w < 320;
  const L=narrow?28:44,R=narrow?30:16,Tp=18,B=30, pw=w-L-R, ph=h-Tp-B;
  const ymin=-0.8, ymax=2.0;
  const xs = j => L + pw*(j/tr.n), ys = v => Tp + ph*(1-(v-ymin)/(ymax-ymin));
  ctx.clearRect(0,0,w,h);
  // axes
  ctx.strokeStyle=COL.rule; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(L,ys(0)); ctx.lineTo(w-R,ys(0)); ctx.stroke();
  ctx.fillStyle=COL.muted; ctx.font='12px "Instrument Sans",sans-serif';
  for(let s=0;s<tr.T;s+=(narrow?1:0.5)){ const x=L+pw*(s/tr.T); ctx.fillText(narrow?(s+' s'):((s*1000)+' ms'), narrow?x-6:x-14, h-8); }
  ctx.save(); ctx.translate(narrow?10:12, Tp+ph/2+28); ctx.rotate(-Math.PI/2); ctx.fillText('evidence', 0,0); ctx.restore();
  // threshold
  ctx.strokeStyle=COL.thr; ctx.setLineDash([7,5]); ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(L,ys(tr.theta)); ctx.lineTo(w-R,ys(tr.theta)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=COL.thr; ctx.font='600 12px "Instrument Sans",sans-serif'; ctx.fillText('Θ(c)', L+4, ys(tr.theta)-6);
  const last = Math.min(upto, tr.hit);
  const ly = [0,1,2,3].map(i=>({i,y:ys(Math.max(ymin,Math.min(ymax,tr.X[i][last])))})).sort((p,q)=>p.y-q.y);
  for(let k=1;k<4;k++) if(ly[k].y-ly[k-1].y<13) ly[k].y=ly[k-1].y+13;
  const labY = {}; ly.forEach(o=>labY[o.i]=o.y);
  for(let i=0;i<4;i++){
    ctx.strokeStyle=HCOL[i]; ctx.globalAlpha = (upto>=tr.hit && i!==tr.win)?0.45:1; ctx.lineWidth = (upto>=tr.hit && i===tr.win)?2.6:1.8;
    ctx.beginPath(); for(let j=0;j<=last;j++){ const x=xs(j),y=ys(Math.max(ymin,Math.min(ymax,tr.X[i][j]))); j?ctx.lineTo(x,y):ctx.moveTo(x,y); } ctx.stroke();
    ctx.globalAlpha=1;
    const y = ys(Math.max(ymin,Math.min(ymax,tr.X[i][last])));
    ctx.fillStyle=HCOL[i]; ctx.font='500 12px "Instrument Sans",sans-serif'; ctx.fillText(LABELS[i], xs(last)+8, labY[i]+4);
  }
  if(upto>=tr.hit){
    const x=xs(tr.hit), y=ys(tr.theta);
    ctx.strokeStyle=COL.ink; ctx.lineWidth=1; ctx.setLineDash([2,3]); ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x,Tp+ph); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=HCOL[tr.win]; ctx.beginPath(); ctx.arc(x,y,6,0,7); ctx.fill();
    ctx.fillStyle=COL.ink; ctx.font='600 12px "Instrument Sans",sans-serif'; ctx.fillText('t* = '+Math.round(tr.hit*tr.dt*1000)+' ms', x+8, Tp+ph-6);
  }
}
function heroRun(){
  heroStop();
  const tr = heroTrial();
  heroShowStimulus(tr);
  const said = () => { const ok = tr.win===tr.target; heroOut.innerHTML = `Choice: <b>${LABELS[tr.win]}</b> after <b>${Math.round(tr.hit*tr.dt*1000)} ms</b>` + (ok?'':' (an error: the stimulus favoured '+LABELS[tr.target]+')'); };
  if(reduce){ heroDraw(tr, tr.n); said(); return; }
  heroOut.textContent = 'Accumulating evidence…';
  let j=0; const step = Math.max(2, Math.round(tr.n/160));
  const tick = () => {
    heroFrame = null; j += step; heroDraw(tr, j);
    if(j < tr.hit){ heroFrame = requestAnimationFrame(tick); }
    else { said(); heroReplay = setTimeout(() => { heroReplay = null; if(autoplay) heroRun(); }, 2600); }
  };
  tick();
}
let autoplay = !reduce;
document.getElementById('heroBtn').addEventListener('click', heroRun);
new IntersectionObserver(es => es.forEach(e => { autoplay = e.isIntersecting && !reduce; })).observe(heroCv);

/* ---------------- PLAYGROUND 1 ---------------- */
const $ = id => document.getElementById(id);
const NDT = 0.30; // non-decision time (s)
function simRace(coh, sig, thr, nTrials, seed, keep){
  const r = rng(seed), dt=0.005, T=3.0, n=Math.round(T/dt);
  const vC = 1.0 + 7.0*coh, vE = 1.0;
  const rts=[], cor=[], paths=[];
  for(let k=0;k<nTrials;k++){
    let a=0,b=0,j=0; const pa=keep&&k<keep?[0]:null, pb=keep&&k<keep?[0]:null;
    for(j=1;j<=n;j++){ a+=vC*dt+sig*Math.sqrt(dt)*gauss(r); b+=vE*dt+sig*Math.sqrt(dt)*gauss(r); if(pa){pa.push(a);pb.push(b);} if(a>=thr||b>=thr) break; }
    rts.push(Math.min(j,n)*dt+NDT); cor.push(a>=b);
    if(pa) paths.push({pa,pb,c:a>=b});
  }
  return {rts,cor,paths,dt,T};
}
const median = a => { if(!a.length) return NaN; const s=[...a].sort((x,y)=>x-y), m=s.length>>1; return s.length%2?s[m]:(s[m-1]+s[m])/2; };
function drawPG(){
  const coh=+$('pgCoh').value, sig=+$('pgSig').value, thr=+$('pgThr').value;
  $('pgCohV').textContent = Math.round(coh*100)+'%'; $('pgSigV').textContent=sig.toFixed(2); $('pgThrV').textContent=thr.toFixed(2);
  const S = simRace(coh,sig,thr,500,1234,24);
  // trajectories
  let {ctx,w,h} = setup($('pgTraj'));
  const sorted=[...S.rts].sort((x,y)=>x-y); const tMax=Math.min(3,Math.max(0.8,sorted[Math.floor(sorted.length*0.97)]-NDT+0.15));
  const L=40,R=10,Tp=12,B=24,pw=w-L-R,ph=h-Tp-B, ymin=-0.9, ymax=thr+0.35;
  const xs=t=>L+pw*(t/tMax), ys=v=>Tp+ph*(1-(Math.max(ymin,Math.min(ymax,v))-ymin)/(ymax-ymin));
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle=COL.rule; ctx.beginPath(); ctx.moveTo(L,ys(0)); ctx.lineTo(w-R,ys(0)); ctx.stroke();
  ctx.fillStyle=COL.muted; ctx.font='11px "Instrument Sans",sans-serif';
  const tstep=tMax>1.6?0.5:0.25; for(let t=0;t<=tMax;t+=tstep) ctx.fillText(t.toFixed(2)+' s', xs(t)-12, h-6);
  ctx.save(); ctx.beginPath(); ctx.rect(L,0,pw,h); ctx.clip();
  S.paths.forEach(p=>{
    [[p.pa,COL.a],[p.pb,COL.b]].forEach(([arr,c])=>{ ctx.strokeStyle=c; ctx.globalAlpha=.42; ctx.lineWidth=1.1; ctx.beginPath(); arr.forEach((v,j)=>{ const x=xs(j*S.dt), y=ys(v); j?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke(); });
    ctx.globalAlpha=1; const j=p.pa.length-1; ctx.fillStyle=p.c?COL.a:COL.b; ctx.beginPath(); ctx.arc(xs(j*S.dt), ys(thr), 2.6, 0, 7); ctx.fill();
  }); ctx.restore();
  ctx.strokeStyle=COL.thr; ctx.lineWidth=2; ctx.setLineDash([6,4]); ctx.beginPath(); ctx.moveTo(L,ys(thr)); ctx.lineTo(w-R,ys(thr)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle=COL.thr; ctx.font='600 11px "Instrument Sans",sans-serif'; ctx.fillText('Θ', L-14, ys(thr)+4);
  // histogram
  ({ctx,w,h} = setup($('pgHist')));
  const HL=40,HR=10,mid=h/2-6, bins=44, bw=(2.6)/bins; const hc=new Array(bins).fill(0), he=new Array(bins).fill(0);
  S.rts.forEach((t,i)=>{ const b=Math.min(bins-1,Math.floor(t/bw)); (S.cor[i]?hc:he)[b]++; });
  const mx=Math.max(1,...hc,...he), bpx=(w-HL-HR)/bins;
  ctx.clearRect(0,0,w,h);
  for(let b=0;b<bins;b++){ const x=HL+b*bpx; const hC=(mid-6)*hc[b]/mx, hE=(mid-6)*he[b]/mx;
    ctx.fillStyle=COL.a; ctx.globalAlpha=.8; ctx.fillRect(x,mid-hC,bpx-1,hC); ctx.fillStyle=COL.b; ctx.fillRect(x,mid,bpx-1,hE); ctx.globalAlpha=1; }
  ctx.strokeStyle=COL.ink; ctx.beginPath(); ctx.moveTo(HL,mid); ctx.lineTo(w-HR,mid); ctx.stroke();
  ctx.fillStyle=COL.muted; ctx.font='11px "Instrument Sans",sans-serif';
  for(let t=0;t<=2.5;t+=0.5) ctx.fillText(t.toFixed(1)+' s', HL+(t/bw)*bpx-10, h-4);
  ctx.fillText('correct', 2, mid-8); ctx.fillText('errors', 2, mid+16);
  // stats
  const rc=S.rts.filter((_,i)=>S.cor[i]), re=S.rts.filter((_,i)=>!S.cor[i]);
  $('stAcc').textContent = Math.round(100*rc.length/S.rts.length)+'%';
  $('stRT').textContent = Math.round(1000*median(S.rts))+' ms';
  $('stRTc').textContent = rc.length? Math.round(1000*median(rc))+' ms':'–';
  $('stRTe').textContent = re.length>4? Math.round(1000*median(re))+' ms':'–';
}
// A redraw simulates hundreds of trials, so coalesce the `input` burst a
// dragged slider emits down to one render per frame.
function coalesce(fn){
  let pending = null;
  return () => { if(pending !== null) return; pending = requestAnimationFrame(() => { pending = null; fn(); }); };
}
const queuePG = coalesce(drawPG);
['pgCoh','pgSig','pgThr'].forEach(id=>$(id).addEventListener('input', queuePG));
$('pgSweep').addEventListener('click', () => {
  const sig=+$('pgSig').value, thr=+$('pgThr').value, cohs=[0.008,0.016,0.032,0.064,0.128,0.256,0.512];
  const acc=[], rt=[];
  cohs.forEach((c,i)=>{ const S=simRace(c,sig,thr,600,99+i,0); acc.push(S.cor.filter(Boolean).length/S.cor.length); rt.push(1000*median(S.rts)); });
  $('sweepOut').hidden=false;
  lineChart($('swAcc'), cohs.map(c=>(c*100).toFixed(1)), [{name:'simulated race',v:acc.map(a=>a*100),c:COL.ink}], {yl:'accuracy (%)', xl:'strength (%)', ymin:40, ymax:100});
  lineChart($('swRT'), cohs.map(c=>(c*100).toFixed(1)), [{name:'simulated race',v:rt,c:COL.ink}], {yl:'median RT (ms)', xl:'strength (%)'});
});

/* ---------------- SURROGATE ---------------- */
let sgSeed = 42;
const sig = x => 1/(1+Math.exp(-x));
function sgCompute(){
  const kappa = Math.round(Math.pow(10, 0.3 + (+$('sgK').value)*2.7)); // ~2 .. ~1000
  const M = +$('sgM').value;
  $('sgKV').textContent = kappa; $('sgMV').textContent = M;
  const r=rng(sgSeed), dt=0.01, T=3.0, n=Math.round(T/dt), thr=1.0, vA=1.4, vB=0.9, s=0.8;
  const soft=new Float64Array(n+1), hard=new Float64Array(n+1); let ex=null;
  for(let m=0;m<M;m++){
    // paper's convention: P_cross(t0) = 0
    let a=0,b=0,prev=0, hit=-1; const tr=m===0?{x:[0],p:[0]}:null;
    for(let j=1;j<=n;j++){
      a+=vA*dt+s*Math.sqrt(dt)*gauss(r); b+=vB*dt+s*Math.sqrt(dt)*gauss(r);
      const p = Math.max(sig(kappa*(a-thr)), sig(kappa*(b-thr)));
      soft[j] += Math.max(p-prev,0); prev=p;
      if(hit<0 && (a>=thr||b>=thr)){ hit=j; hard[j]+=1; }
      if(tr){ tr.x.push(Math.max(a,b)); tr.p.push(p); }
    }
    if(tr) ex={...tr,hit};
  }
  const ss=soft.reduce((x,y)=>x+y,0)+1e-9, hs=hard.reduce((x,y)=>x+y,0)+1e-9;
  for(let j=0;j<=n;j++){ soft[j]/=ss; hard[j]/=hs; }
  // total variation on 5-bin-coarsened grid
  let tv=0, ms=0, mh=0; const g=5;
  for(let j=0;j<=n;j+=g){ let a=0,b=0; for(let k=j;k<Math.min(n+1,j+g);k++){a+=soft[k];b+=hard[k];} tv+=Math.abs(a-b); }
  for(let j=0;j<=n;j++){ ms+=soft[j]*j*dt; mh+=hard[j]*j*dt; }
  $('sgTV').textContent=(tv/2).toFixed(3); $('sgShift').textContent=(Math.round((ms-mh)*1000)>0?'+':'')+Math.round((ms-mh)*1000)+' ms';
  // draw trajectory
  let {ctx,w,h}=setup($('sgTraj'));
  const L=40,R=40,Tp=12,B=22,pw=w-L-R,ph=h-Tp-B,tMax=2.0,ymin=-0.3,ymax=1.8;
  const xs=t=>L+pw*(t/tMax), ys=v=>Tp+ph*(1-(Math.max(ymin,Math.min(ymax,v))-ymin)/(ymax-ymin));
  ctx.clearRect(0,0,w,h);
  // soft band: gradient of sigma around threshold
  const band = Math.min(3, 4/kappa);
  const gy0=ys(thr+band), gy1=ys(thr-band);
  const grd=ctx.createLinearGradient(0,gy0,0,gy1); grd.addColorStop(0,'rgba(201,138,0,0.0)'); grd.addColorStop(0.5,'rgba(201,138,0,0.28)'); grd.addColorStop(1,'rgba(201,138,0,0.0)');
  ctx.fillStyle=grd; ctx.fillRect(L,gy0,pw,gy1-gy0);
  ctx.strokeStyle=COL.thr; ctx.lineWidth=1.5; ctx.setLineDash([6,4]); ctx.beginPath(); ctx.moveTo(L,ys(thr)); ctx.lineTo(L+pw,ys(thr)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle=COL.ink; ctx.lineWidth=1.6; ctx.beginPath(); ex.x.forEach((v,j)=>{ if(j*dt>tMax) return; const x=xs(j*dt),y=ys(v); j?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
  // P_cross curve on right axis (0..1)
  const yp=p=>Tp+ph*(1-p);
  ctx.strokeStyle=COL.b; ctx.lineWidth=1.6; ctx.beginPath(); ex.p.forEach((p,j)=>{ if(j*dt>tMax) return; const x=xs(j*dt),y=yp(p); j?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
  if(ex.hit>0 && ex.hit*dt<=tMax){ ctx.fillStyle=COL.ink; ctx.beginPath(); ctx.arc(xs(ex.hit*dt),ys(thr),4,0,7); ctx.fill(); }
  ctx.fillStyle=COL.muted; ctx.font='11px "Instrument Sans",sans-serif';
  for(let t=0;t<=2;t+=0.5) ctx.fillText(t.toFixed(1)+' s', xs(t)-10, h-5);
  ctx.fillText('max X', 2, Tp+10); ctx.fillStyle=COL.b; ctx.fillText('P cross', w-R+2, Tp+10); ctx.fillText('1', w-R+4, yp(1)+14); ctx.fillText('0', w-R+4, yp(0));
  // density
  ({ctx,w,h}=setup($('sgDens')));
  const DL=40,DR=12,DT=10,DB=22,dw=w-DL-DR,dh=h-DT-DB;
  const agg=(arr)=>{ const out=[]; for(let j=0;j<=n;j+=g){ let a=0; for(let k=j;k<Math.min(n+1,j+g);k++) a+=arr[k]; out.push(a);} return out; };
  const S2=agg(soft), H2=agg(hard), mx=Math.max(...S2,...H2)*1.1, nb=S2.length, bx=dw/(nb*(tMax/T)) ;
  const X2=i=>DL+(i*g*dt/tMax)*dw, Y2=v=>DT+dh*(1-v/mx);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#9AA6B8';
  H2.forEach((v,i)=>{ const x=X2(i); if(x>DL+dw) return; ctx.fillRect(x, Y2(v), Math.max(1,X2(1)-X2(0)-1), DT+dh-Y2(v)); });
  ctx.strokeStyle=COL.b; ctx.lineWidth=2.2; ctx.beginPath(); S2.forEach((v,i)=>{ const x=X2(i)+(X2(1)-X2(0))/2; if(x>DL+dw) return; const y=Y2(v); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
  ctx.strokeStyle=COL.rule; ctx.beginPath(); ctx.moveTo(DL,DT+dh); ctx.lineTo(DL+dw,DT+dh); ctx.stroke();
  ctx.fillStyle=COL.muted; ctx.font='11px "Instrument Sans",sans-serif';
  for(let t=0;t<=2;t+=0.5) ctx.fillText(t.toFixed(1)+' s', DL+(t/tMax)*dw-10, h-5);
  ctx.fillText('density', 2, DT+10);
}
const queueSG = coalesce(sgCompute);
['sgK','sgM'].forEach(id=>$(id).addEventListener('input', queueSG));
$('sgNew').addEventListener('click', ()=>{ sgSeed+=17; sgCompute(); });

/* ---------------- SVG CHARTS ---------------- */
const NS='http://www.w3.org/2000/svg';
function el(tag, attrs, parent){ const e=document.createElementNS(NS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); if(parent) parent.appendChild(e); return e; }
function lineChart(host, xlabels, series, o={}){
  host.innerHTML=''; const W=420,H=250,L=46,R=12,T=12,B=52;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':(o.yl||'')+' by '+(o.xl||'')},host);
  const all=series.flatMap(s=>s.v); let ymin=o.ymin??Math.min(...all), ymax=o.ymax??Math.max(...all);
  if(o.ymin===undefined){ const pad=(ymax-ymin)*0.12||1; ymin-=pad; ymax+=pad; }
  const xs=i=>L+(W-L-R)*(i/(xlabels.length-1)), ys=v=>T+(H-T-B)*(1-(v-ymin)/(ymax-ymin));
  for(let k=0;k<=4;k++){ const v=ymin+(ymax-ymin)*k/4, y=ys(v); el('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:COL.rule,'stroke-width':1},svg); const t=el('text',{x:L-6,y:y+4,'text-anchor':'end'},svg); t.textContent=Math.round(v); }
  xlabels.forEach((lb,i)=>{ const t=el('text',{x:xs(i),y:H-B+16,'text-anchor':'middle'},svg); t.textContent=lb; });
  if(o.xl){ const t=el('text',{x:(L+W-R)/2,y:H-B+32,'text-anchor':'middle'},svg); t.textContent=o.xl; }
  series.forEach(s=>{
    el('polyline',{points:s.v.map((v,i)=>xs(i)+','+ys(v)).join(' '),fill:'none',stroke:s.c,'stroke-width':s.w||2.2,'stroke-dasharray':s.dash||'', opacity:s.op||1},svg);
    s.v.forEach((v,i)=>el('circle',{cx:xs(i),cy:ys(v),r:s.w?3.4:2.6,fill:s.c,opacity:s.op||1},svg));
  });
  // legend
  let lx=L; series.forEach(s=>{ el('line',{x1:lx,x2:lx+16,y1:H-8,y2:H-8,stroke:s.c,'stroke-width':2.4,'stroke-dasharray':s.dash||''},svg); const t=el('text',{x:lx+20,y:H-4},svg); t.textContent=s.name; lx+=20+s.name.length*6.2+14; });
}
function barChart(host, groups, series, o={}){
  host.innerHTML=''; const W=860,H=300,L=46,R=12,T=14,B=56;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':o.label||'bar chart'},host);
  const ymax=o.ymax||Math.max(...series.flatMap(s=>s.v))*1.08;
  const bars=series.filter(s=>!s.line); const ys=v=>T+(H-T-B)*(1-v/ymax), gw=(W-L-R)/groups.length, bw=Math.min(24,(gw-24)/bars.length);
  for(let k=0;k<=4;k++){ const v=ymax*k/4, y=ys(v); el('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:COL.rule},svg); const t=el('text',{x:L-6,y:y+4,'text-anchor':'end'},svg); t.textContent=Math.round(v); }
  groups.forEach((g,gi)=>{
    const x0=L+gi*gw+(gw-bw*bars.length)/2;
    bars.forEach((s,si)=>{ const v=s.v[gi]; el('rect',{x:x0+si*bw,y:ys(v),width:bw-2,height:ys(0)-ys(v),fill:s.c,rx:2},svg); });
    const hum=series.find(s=>s.line); if(hum){ const y=ys(hum.v[gi]); el('line',{x1:x0-6,x2:x0+bw*bars.length+4,y1:y,y2:y,stroke:COL.ink,'stroke-width':2,'stroke-dasharray':'5 3'},svg); }
    const t=el('text',{x:L+gi*gw+gw/2,y:H-B+18,'text-anchor':'middle','font-weight':600,style:'fill:'+COL.ink},svg); t.textContent=g;
  });
  let lx=L; series.forEach(s=>{ if(s.line){ el('line',{x1:lx,x2:lx+16,y1:H-10,y2:H-10,stroke:COL.ink,'stroke-width':2,'stroke-dasharray':'5 3'},svg);} else el('rect',{x:lx,y:H-16,width:12,height:12,fill:s.c,rx:2},svg); const t=el('text',{x:lx+18,y:H-6},svg); t.textContent=s.name; lx+=24+s.name.length*6.3+10; });
}

// Data from the paper (Tables 1, 5, 7, 3)
const coh=['0.8','1.6','3.2','6.4','12.8','25.6','51.2'];
const GREY='#9AA6B8', TEAL=COL.c, PURP=COL.d;
function drawResults(){
  lineChart($('cRdmAcc'), coh, [
    {name:'Human',v:[49.1,53.3,56.7,60.7,71.1,93.0,98.7],c:COL.human||COL.a,w:3},
    {name:'NRM',v:[51.9,57.8,64.8,82.0,99.5,100,100],c:COL.b,w:3},
    {name:'RTify',v:[50.3,51.7,51.4,53.9,58.5,55.7,62.1],c:TEAL,dash:'4 3'},
    {name:'DDM',v:[51.7,51.5,47.1,51.0,48.6,50.6,50.2],c:GREY,dash:'4 3'}], {xl:'coherence (%)', ymin:40, ymax:100});
  lineChart($('cRdmRT'), coh, [
    {name:'Human',v:[714,703,683,666,642,568,503],c:COL.a,w:3},
    {name:'NRM',v:[672,672,672,662,632,582,532],c:COL.b,w:3},
    {name:'RTify',v:[613,680,640,613,627,587,560],c:TEAL,dash:'4 3'},
    {name:'DDM',v:[624,625,623,624,624,625,627],c:GREY,dash:'4 3'}], {xl:'coherence (%)', ymin:480, ymax:740});
  lineChart($('cScAcc'), ['0.2','0.4','0.6','0.8','1.0'], [
    {name:'Human',v:[73.8,81.8,82.5,88.0,91.3],c:COL.a,w:3},
    {name:'NRM',v:[75.2,83.2,85.7,79.0,94.0],c:COL.b,w:3},
    {name:'RTify',v:[46.7,75.0,79.6,86.1,96.8],c:TEAL,dash:'4 3'},
    {name:'cRNN',v:[37.9,53.6,62.7,56.6,93.7],c:PURP,dash:'4 3'},
    {name:'DDM',v:[52.3,47.9,49.9,54.0,49.6],c:GREY,dash:'4 3'}], {xl:'discriminability', ymin:30, ymax:100});
  // kappa as horizontal bars via barChart-like simple svg
  const host=$('cScK'); host.innerHTML='';
  const rows=[['Human (NC)',0.679,COL.a],['NRM',0.666,COL.b],['RTify',0.562,TEAL],['cRNN',0.240,PURP],['LBA',0.033,GREY],['DDM',0.013,GREY],['LNR',-0.015,GREY]];
  const W=420,H=250,L=86,R=40, rh=(H-20)/rows.length, x0=L, sc=(W-L-R)/0.75;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':"Cohen's kappa per model"},host);
  rows.forEach(([n,v,c],i)=>{ const y=10+i*rh; const t=el('text',{x:L-8,y:y+rh/2+4,'text-anchor':'end',style:'fill:'+COL.ink},svg); t.textContent=n;
    const w=Math.max(1,Math.abs(v)*sc); el('rect',{x:v>=0?x0:x0-w,y:y+4,width:w,height:rh-9,fill:c,rx:2},svg);
    const tv=el('text',{x:x0+Math.max(0,v)*sc+6,y:y+rh/2+4},svg); tv.textContent=v.toFixed(3); });
  el('line',{x1:x0,x2:x0,y1:6,y2:H-6,stroke:COL.ink},svg);
  barChart($('cSat'), ['Color & Blur','Color & Gray','Gray & Noise'], [
    {name:'NRM',v:[30.67,24.32,29.74],c:COL.b},
    {name:'CNet-P',v:[31.72,27.18,35.48],c:'#7FA3E0'},
    {name:'CNet-S',v:[33.58,26.57,30.84],c:'#4F7FD0'},
    {name:'ConvRNN',v:[50.38,50.72,32.90],c:PURP},
    {name:'MSDNet',v:[57.74,31.73,65.96],c:'#E3A857'},
    {name:'SCAN',v:[36.09,39.50,47.51],c:TEAL},
    {name:'Human noise ceiling',v:[24.34,21.62,23.08],line:true}], {ymax:80, label:'RMSE per condition and model'});
}

// tabs
// each tablist switches only its own panels
document.querySelectorAll('.tabs').forEach(list=>{
  const tabs=[...list.querySelectorAll('.tab')];
  tabs.forEach(t=>t.addEventListener('click',()=>{ tabs.forEach(x=>{ x.setAttribute('aria-selected', x===t); $(x.getAttribute('aria-controls')).hidden = x!==t; }); }));
  tabs.forEach((t,i)=>t.addEventListener('keydown',e=>{ if(e.key==='ArrowRight'||e.key==='ArrowLeft'){ const n=tabs[(i+(e.key==='ArrowRight'?1:tabs.length-1))%tabs.length]; n.focus(); n.click(); } }));
});

$('copyBib').addEventListener('click', async () => { try{ await navigator.clipboard.writeText($('bibText').textContent); $('copyBib').textContent='Copied'; }catch(e){ $('copyBib').textContent='Select and copy the text above'; } setTimeout(()=>$('copyBib').textContent='Copy BibTeX',1800); });

// Each panel is independent: a failure in one must not stop the others from
// rendering or leave their controls dead.
function guard(label, fn){ try{ fn(); }catch(e){ console.error('[nrm] '+label+' failed', e); } }
function all(){ guard('playground', drawPG); guard('surrogate', sgCompute); }
guard('results', drawResults);
guard('hero', heroRun);
all();
let rz; window.addEventListener('resize',()=>{ clearTimeout(rz); rz=setTimeout(all,150); });
if(document.fonts) document.fonts.ready.then(all);
})();
