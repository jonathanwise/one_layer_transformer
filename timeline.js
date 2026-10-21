// timeline.js — model-run selector and training-time slider shared by the viewers.
// Usage:
//   Timeline.init({ container: '#timeline', modes: ['original', ...] or ['c3'], priority: () => currentMode,
//                   onIndex(index), onStart(step), onData(step, mode, data, complete), onError(msg) });
// The run comes from the page's ?run=<dir> query parameter (default: the first entry of runs.json, the
// single-snapshot final model). Each run directory holds index.json (with the run's frequencies K) and
// <step>.bin snapshots; model_worker.js recomputes the requested modes from the raw weights of the step.
// A single-snapshot run shows no slider.
const Timeline = (() => {
  const VERSION = 5;   // bump together with the ?v= on the script tags when the JS changes
  let index = null, runs = [], runDir = null, idx = 0, worker = null, reqId = 0, opts = null, playing = false, playTimer = null;
  let pending = null, busy = false, received = 0, el = {};
  const ML = 38, MR = 34;   // plot margins inside the chart canvases (used by drawCharts and click picking)
  const PALETTE = ['#e94560', '#3b82f6', '#22c55e', '#ffb347', '#c084fc', '#2dd4bf'];
  const q = s => document.querySelector(s);

  function build(container) {
    const root = typeof container === 'string' ? q(container) : container;
    const single = index.single;
    // Switch links: when the viewer sits inside the tabbed index.html, reload that page (same tab kept via its hash).
    let base = '';
    try { if (window.top !== window) base = window.top.location.pathname + '{Q}' + window.top.location.hash; } catch (e) { base = ''; }
    const href = dir => base ? base.replace('{Q}', `?run=${encodeURIComponent(dir)}`) : `?run=${encodeURIComponent(dir)}`;
    const switcher = runs.map(r => {
      const cur = r.dir === runDir;
      const label = `${r.name} · freqs ${r.K.join(', ')}${r.steps > 1 ? ` · ${r.steps} snapshots` : ''}`;
      return cur ? `<span class="tl-run tl-cur">${label}</span>` : `<a class="tl-run" href="${href(r.dir)}" target="_top">${label}</a>`;
    }).join('');
    root.innerHTML = `
      <style>
        .tl { background: #16213e; border-top: 1px solid #333; border-bottom: 1px solid #333; padding: 6px 20px; font-family: 'Menlo','Consolas',monospace; font-size: 12px; color: #aaa; }
        .tl-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .tl button { background: #0f3460; color: #e0e0e0; border: 1px solid #444; padding: 2px 9px; cursor: pointer; border-radius: 3px; font-family: inherit; font-size: 12px; }
        .tl button:hover { background: #1a5276; }
        .tl input[type=range] { flex: 1; min-width: 200px; }
        .tl .tl-step { color: #e94560; font-weight: bold; min-width: 150px; }
        .tl .tl-status { color: #ffb347; min-width: 90px; }
        .tl canvas { display: block; width: 100%; cursor: crosshair; }
        .tl .tl-chart { height: 78px; }
        .tl .tl-freq { height: 46px; }
        .tl .tl-legend { color: #777; font-size: 11px; }
        .tl .tl-legend b { font-weight: normal; }
        .tl .tl-run { padding: 2px 8px; border-radius: 3px; border: 1px solid #444; color: #8ab4f8; text-decoration: none; margin-right: 6px; }
        .tl .tl-run.tl-cur { background: #e94560; color: #fff; border-color: #e94560; }
        .tl a.tl-run:hover { background: #1a5276; }
        .tl [hidden] { display: none !important; }
        .tl .tl-note { color: #ffb347; }
      </style>
      <div class="tl">
        <div class="tl-row"><span style="color:#888">model:</span> ${switcher} <span class="tl-status"></span></div>
        ${single ? `<div class="tl-row tl-note">this run has a single snapshot (no training timeline) · switch to a run with snapshots above to scrub through training</div>` : ''}
        <div class="tl-row tl-slider" ${single ? 'hidden' : ''}>
          <span style="color:#888">training time</span>
          <button class="tl-prev" title="[">◀</button>
          <button class="tl-play" title="space">▶ play</button>
          <button class="tl-next" title="]">▶</button>
          <span class="tl-step"></span>
          <input type="range" class="tl-range" min="0" max="0" value="0">
          <span class="tl-legend">keys: <b>[</b> <b>]</b> step · <b>space</b> play · click the chart to jump</span>
        </div>
        <canvas class="tl-chart" ${single ? 'hidden' : ''}></canvas>
        <canvas class="tl-freq" ${single ? 'hidden' : ''}></canvas>
        <div class="tl-legend tl-caption"></div>
      </div>`;
    el = { root, step: root.querySelector('.tl-step'), range: root.querySelector('.tl-range'), status: root.querySelector('.tl-status'),
           chart: root.querySelector('.tl-chart'), freq: root.querySelector('.tl-freq'), play: root.querySelector('.tl-play'), caption: root.querySelector('.tl-caption') };
    if (single) return;
    el.range.max = index.steps.length - 1;
    el.range.addEventListener('input', e => setIndex(+e.target.value));
    root.querySelector('.tl-prev').addEventListener('click', () => setIndex(idx - 1));
    root.querySelector('.tl-next').addEventListener('click', () => setIndex(idx + 1));
    el.play.addEventListener('click', togglePlay);
    for (const c of [el.chart, el.freq]) {
      const pick = e => { const r = c.getBoundingClientRect(); const x = (e.clientX - r.left - ML) / (r.width - ML - MR); setIndex(Math.round(x * (index.steps.length - 1))); };
      c.addEventListener('mousedown', e => { pick(e); c._drag = true; });
      c.addEventListener('mousemove', e => { if (c._drag && e.buttons) pick(e); });
      c.addEventListener('mouseup', () => { c._drag = false; });
      c.addEventListener('mouseleave', () => { c._drag = false; });
    }
    document.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT') return;
      if (e.key === '[') { setIndex(idx - 1); e.preventDefault(); }
      else if (e.key === ']') { setIndex(idx + 1); e.preventDefault(); }
      else if (e.key === ' ') { togglePlay(); e.preventDefault(); }
    });
    window.addEventListener('resize', drawCharts);
  }

  function togglePlay() {
    playing = !playing;
    el.play.textContent = playing ? '❚❚ pause' : '▶ play';
    if (playing) {
      if (idx >= index.steps.length - 1) setIndex(0);
      playTimer = setInterval(() => { if (!busy) { if (idx >= index.steps.length - 1) togglePlay(); else setIndex(idx + 1); } }, 60);
    } else { clearInterval(playTimer); playTimer = null; }
  }

  function caption() {
    const r = index.summary[idx];
    const lam = `|Λ| = ${r.Lambda.map(z => Math.hypot(z[0], z[1]).toFixed(1)).join(', ')} (k = ${index.K.join(', ')})`;
    if (index.single) return `accuracy on all 1849 pairs ${(100 * r.train_acc).toFixed(1)}% · loss ${r.train_loss.toExponential(2)} · ${lam}`;
    return `<span style="color:#ffb347">train</span> loss ${r.train_loss.toExponential(2)} acc ${(100 * r.train_acc).toFixed(1)}% · ` +
      `<span style="color:#8ab4f8">test</span> loss ${r.test_loss.toExponential(2)} acc ${(100 * r.test_acc).toFixed(1)}% · ` +
      `strip: W_E energy by frequency (rows k = 1…21; ${index.K.map((k, i) => `<span style="color:${PALETTE[i % PALETTE.length]}">k=${k}</span>`).join(', ')}) · ${lam}`;
  }

  function drawCharts() {
    el.caption.innerHTML = caption();
    if (index.single) return;
    const S = index.summary, n = S.length;
    {
      const c = el.chart, W = c.clientWidth || 800, H = 78;
      c.width = W * devicePixelRatio; c.height = H * devicePixelRatio; c.style.height = H + 'px';
      const ctx = c.getContext('2d'); ctx.scale(devicePixelRatio, devicePixelRatio); ctx.clearRect(0, 0, W, H);
      const ml = ML, mr = MR, mt = 4, mb = 12, pw = W - ml - mr, ph = H - mt - mb;
      const X = i => ml + pw * i / Math.max(n - 1, 1);
      let lmin = Infinity, lmax = -Infinity;
      for (const r of S) for (const v of [r.train_loss, r.test_loss]) { if (v > 0) { lmin = Math.min(lmin, v); lmax = Math.max(lmax, v); } }
      lmin = Math.log10(Math.max(lmin, 1e-4)); lmax = Math.log10(lmax);
      const YL = v => mt + ph * (1 - (Math.log10(Math.max(v, 1e-4)) - lmin) / (lmax - lmin || 1));
      const YA = v => mt + ph * (1 - v);
      ctx.strokeStyle = '#2a2a45'; ctx.lineWidth = 1;
      for (let e = Math.ceil(lmin); e <= Math.floor(lmax); e++) { const y = YL(Math.pow(10, e)); ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(W - mr, y); ctx.stroke(); ctx.fillStyle = '#666'; ctx.font = '9px monospace'; ctx.textAlign = 'right'; ctx.fillText('1e' + e, ml - 3, y + 3); }
      const line = (key, color, Y, dash) => { ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.setLineDash(dash || []); ctx.beginPath(); S.forEach((r, i) => i ? ctx.lineTo(X(i), Y(r[key])) : ctx.moveTo(X(i), Y(r[key]))); ctx.stroke(); ctx.setLineDash([]); };
      line('train_acc', 'rgba(255,179,71,0.45)', YA, [3, 3]); line('test_acc', 'rgba(138,180,248,0.45)', YA, [3, 3]);
      line('train_loss', '#ffb347', YL); line('test_loss', '#8ab4f8', YL);
      ctx.fillStyle = '#666'; ctx.textAlign = 'left'; ctx.fillText('acc 1', W - mr + 3, YA(1) + 7); ctx.fillText('acc 0', W - mr + 3, YA(0));
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(idx), mt); ctx.lineTo(X(idx), H - mb); ctx.stroke();
      ctx.fillStyle = '#888'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
      for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 8))) ctx.fillText(index.steps[i], X(i), H - 2);
    }
    {
      const c = el.freq, W = c.clientWidth || 800, H = 46;
      c.width = W * devicePixelRatio; c.height = H * devicePixelRatio; c.style.height = H + 'px';
      const ctx = c.getContext('2d'); ctx.scale(devicePixelRatio, devicePixelRatio); ctx.clearRect(0, 0, W, H);
      const ml = ML, mr = MR, pw = W - ml - mr, nf = S[0].we.length - 1;
      const rh = H / nf, cw = pw / n;
      let vmax = 0; for (const r of S) for (let k = 1; k <= nf; k++) vmax = Math.max(vmax, r.we[k]);
      const colOf = k => { const i = index.K.indexOf(k); return i < 0 ? [200, 200, 200] : PALETTE[i % PALETTE.length].match(/\w\w/g).map(h => parseInt(h, 16)); };
      for (let i = 0; i < n; i++) for (let k = 1; k <= nf; k++) {
        const t = Math.sqrt(S[i].we[k] / vmax), col = colOf(k);
        ctx.fillStyle = `rgb(${col.map(v => Math.round(v * t + 18 * (1 - t))).join(',')})`;
        ctx.fillRect(ml + i * cw, (k - 1) * rh, Math.ceil(cw) + 0.5, rh + 0.5);
      }
      ctx.fillStyle = '#666'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
      for (const k of [...new Set([1, ...index.K, nf])]) ctx.fillText('k=' + k, ml - 3, (k - 0.5) * rh + 3);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; const X = ml + pw * idx / Math.max(n - 1, 1);
      ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, H); ctx.stroke();
    }
  }

  function setIndex(i) {
    if (index.single) return;
    i = Math.max(0, Math.min(index.steps.length - 1, i));
    if (i === idx && !busy) return;
    idx = i;
    el.range.value = idx;
    el.step.textContent = `step ${index.steps[idx]}  (${idx + 1}/${index.steps.length})`;
    drawCharts();
    request();
  }

  function request() {
    if (busy) { pending = idx; return; }
    busy = true; pending = null; received = 0;
    const step = index.steps[idx];
    reqId++;
    const modes = opts.modes.slice();
    const pri = opts.priority && opts.priority();
    if (pri && modes.includes(pri)) { modes.splice(modes.indexOf(pri), 1); modes.unshift(pri); }
    el.status.textContent = 'computing…';
    if (opts.onStart) opts.onStart(step);
    worker.postMessage({ type: 'compute', id: reqId, step, modes, summary: index.summary[idx] });
  }

  function failed(message) {
    busy = false;
    el.status.textContent = 'error: ' + message;
    console.error('timeline worker:', message);
    if (opts.onError) opts.onError(message);
    if (pending !== null) { const p = pending; pending = null; idx = p; request(); }
  }
  function onWorkerMessage(e) {
    const m = e.data;
    if (m.type === 'error') { failed(m.message); return; }
    if (m.id !== reqId) return;
    received++;
    opts.onData(m.step, m.mode, m.data, received === opts.modes.length);
    if (received >= opts.modes.length) {
      busy = false; el.status.textContent = '';
      if (pending !== null) { const p = pending; pending = null; if (p !== idx) idx = p; request(); }
    }
  }

  async function init(o) {
    try { return await initInner(o); }
    catch (err) { console.error(err); const root = typeof o.container === 'string' ? q(o.container) : o.container; if (root) root.innerHTML = `<div style="color:#e94560;padding:8px 20px;font-family:monospace">timeline failed to load: ${err.message} — run export_snapshots.py first?</div>`; if (o.onError) o.onError(String(err)); }
  }
  async function initInner(o) {
    opts = o;
    try { runs = await (await fetch('runs.json')).json(); } catch (err) { runs = []; }
    const params = new URLSearchParams(location.search);
    runDir = params.get('run') || (runs.length ? runs[0].dir : 'snapshots_final');
    if (!runs.some(r => r.dir === runDir)) runs.push({ dir: runDir, name: runDir, K: [], steps: 0 });
    index = await (await fetch(`${runDir}/index.json`)).json();
    const me = runs.find(r => r.dir === runDir); me.name = index.name; me.K = index.K; me.steps = index.steps.length;
    build(o.container);
    if (o.onIndex) o.onIndex(index);
    worker = new Worker('model_worker.js?v=' + VERSION);
    worker.onmessage = onWorkerMessage;
    worker.onerror = e => { failed((e && e.message) || 'worker failed to load (try a hard refresh: Ctrl+Shift+R)'); };
    worker.postMessage({ type: 'init', base: runDir, index: { order: index.order, shapes: index.shapes, K: index.K } });
    idx = index.steps.length - 1;
    if (!index.single) { el.range.value = idx; el.step.textContent = `step ${index.steps[idx]}  (${idx + 1}/${index.steps.length})`; }
    drawCharts();
    request();
    return index;
  }

  function refresh() { if (index) request(); }   // recompute the current step (e.g. after the page changed its mode list)
  return { init, setIndex, refresh, PALETTE, get index() { return index; }, get idx() { return idx; }, get K() { return index ? index.K : []; } };
})();
