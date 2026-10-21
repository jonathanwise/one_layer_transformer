// model_js.js — the one-layer modular-addition transformer, evaluated in JavaScript.
// Loaded by model_worker.js (importScripts) so the viewers can recompute everything from
// raw weights of any training snapshot. Mirrors model.py / attention.py / mlp.py exactly.
const ModelJS = (() => {
  const P = 43, D = 128, H = 4, DH = 32, DM = 512, V = 44, N = P * P;

  // ---- float16 decode via lookup table ----
  let F16 = null;
  function f16table() {
    if (F16) return F16;
    F16 = new Float32Array(65536);
    for (let u = 0; u < 65536; u++) {
      const s = (u & 0x8000) ? -1 : 1, e = (u >> 10) & 0x1f, f = u & 0x3ff;
      F16[u] = e === 0 ? s * Math.pow(2, -14) * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return F16;
  }
  function decode(buf, index) {
    const T = f16table(), u = new Uint16Array(buf), sd = {};
    let off = 0;
    for (const k of index.order) {
      const n = index.shapes[k].reduce((a, b) => a * b, 1);
      const arr = new Float32Array(n);
      for (let i = 0; i < n; i++) arr[i] = T[u[off + i]];
      off += n; sd[k] = arr;
    }
    return sd;
  }

  // ---- DFT tables ----
  const TW_C = new Float64Array(P * P), TW_S = new Float64Array(P * P);   // e^{-2πi k n / P}
  for (let k = 0; k < P; k++) for (let n = 0; n < P; n++) { const t = -2 * Math.PI * k * n / P; TW_C[k * P + n] = Math.cos(t); TW_S[k * P + n] = Math.sin(t); }

  // Keep only the listed DFT frequencies (k and P−k both expected in `keep`) of the 43 residue rows of W_E.
  function ablateWE(WE, keep) {
    const out = new Float32Array(WE);
    const ks = Array.from(keep);
    for (let d = 0; d < D; d++) {
      const Xr = new Float64Array(P), Xi = new Float64Array(P);
      for (const k of ks) { let sr = 0, si = 0; for (let a = 0; a < P; a++) { const v = WE[a * D + d]; sr += v * TW_C[k * P + a]; si += v * TW_S[k * P + a]; } Xr[k] = sr; Xi[k] = si; }
      for (let a = 0; a < P; a++) {
        let v = 0;
        for (const k of ks) v += Xr[k] * TW_C[k * P + a] + Xi[k] * TW_S[k * P + a];   // Re(X e^{+2πi k a/P}); TW_S holds −sin
        out[a * D + d] = v / P;
      }
    }
    return out;
  }

  // ---- forward pass for all (a, b) inputs, answer position only ----
  function forward(sd, WE) {
    WE = WE || sd.W_E;
    const Wp = sd.W_p, WQ = sd.W_Q, WK = sd.W_K, WV = sd.W_V, WO = sd.W_O, W_i = sd.W_i, b_i = sd.b_i, W_o = sd.W_o, b_o = sd.b_o, W_U = sd.W_U;
    const x0 = (p, t) => { const v = new Float32Array(D); for (let d = 0; d < D; d++) v[d] = WE[t * D + d] + Wp[p * D + d]; return v; };
    const xq = x0(2, P);                                     // answer position holds the "?" token
    // per head: query at the answer position, and scores/OV outputs for every (position, token) it can attend to
    const S = [], VO = [];
    for (let h = 0; h < H; h++) {
      const q = new Float32Array(DH);
      for (let d = 0; d < D; d++) { const xv = xq[d]; if (!xv) continue; const o = h * D * DH + d * DH; for (let e = 0; e < DH; e++) q[e] += xv * WQ[o + e]; }
      S.push([new Float32Array(P), new Float32Array(P), new Float32Array(P + 1)]);
      VO.push([new Array(P), new Array(P), new Array(P + 1)]);
      const slots = [[0, P], [1, P], [2, 1]];
      for (const [p, cnt] of slots) for (let ti = 0; ti < cnt; ti++) {
        const t = p === 2 ? P : ti;
        const x = x0(p, t);
        const k = new Float32Array(DH), v = new Float32Array(DH);
        for (let d = 0; d < D; d++) { const xv = x[d]; if (!xv) continue; const o = h * D * DH + d * DH; for (let e = 0; e < DH; e++) { k[e] += xv * WK[o + e]; v[e] += xv * WV[o + e]; } }
        let s = 0; for (let e = 0; e < DH; e++) s += q[e] * k[e];
        const vo = new Float32Array(D);
        for (let e = 0; e < DH; e++) { const ve = v[e]; if (!ve) continue; const o = h * DH * D + e * D; for (let d = 0; d < D; d++) vo[d] += ve * WO[o + d]; }
        S[h][p][t] = s; VO[h][p][t] = vo;
      }
    }
    const x1 = new Float32Array(N * D), attn = new Float32Array(N * H * 3);
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const i = a * P + b, row = i * D;
      for (let d = 0; d < D; d++) x1[row + d] = xq[d];
      for (let h = 0; h < H; h++) {
        const s0 = S[h][0][a], s1 = S[h][1][b], s2 = S[h][2][P];
        const m = Math.max(s0, s1, s2);
        const e0 = Math.exp(s0 - m), e1 = Math.exp(s1 - m), e2 = Math.exp(s2 - m), z = e0 + e1 + e2;
        const A0 = e0 / z, A1 = e1 / z, A2 = e2 / z;
        attn[(i * H + h) * 3] = A0; attn[(i * H + h) * 3 + 1] = A1; attn[(i * H + h) * 3 + 2] = A2;
        const v0 = VO[h][0][a], v1 = VO[h][1][b], v2 = VO[h][2][P];
        for (let d = 0; d < D; d++) x1[row + d] += A0 * v0[d] + A1 * v1[d] + A2 * v2[d];
      }
    }
    // MLP
    const pre = new Float32Array(N * DM), hid = new Float32Array(N * DM);
    for (let i = 0; i < N; i++) {
      const row = i * DM, xrow = i * D;
      for (let j = 0; j < DM; j++) pre[row + j] = b_i[j];
      for (let d = 0; d < D; d++) { const xv = x1[xrow + d]; if (!xv) continue; const o = d * DM; for (let j = 0; j < DM; j++) pre[row + j] += xv * W_i[o + j]; }
      for (let j = 0; j < DM; j++) hid[row + j] = pre[row + j] > 0 ? pre[row + j] : 0;
    }
    // output: logits = hid (W_o W_U) + b_o W_U
    const WoU = new Float32Array(DM * V);
    for (let j = 0; j < DM; j++) for (let d = 0; d < D; d++) { const w = W_o[j * D + d]; if (!w) continue; const o = d * V, oo = j * V; for (let c = 0; c < V; c++) WoU[oo + c] += w * W_U[o + c]; }
    const bU = new Float32Array(V);
    for (let d = 0; d < D; d++) { const w = b_o[d]; const o = d * V; for (let c = 0; c < V; c++) bU[c] += w * W_U[o + c]; }
    const logits = new Float32Array(N * V);                 // no skip connection: logits = (hid W_o + b_o) W_U
    for (let i = 0; i < N; i++) {
      const o = i * V, hrow = i * DM;
      for (let c = 0; c < V; c++) logits[o + c] = bU[c];
      for (let j = 0; j < DM; j++) { const hv = hid[hrow + j]; if (!hv) continue; const oo = j * V; for (let c = 0; c < V; c++) logits[o + c] += hv * WoU[oo + c]; }
    }
    return { x1, pre, hid, logits, bU, WoU, attn };
  }

  function accuracy(logits) {
    let ok = 0;
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const o = (a * P + b) * V; let best = 0;
      for (let c = 1; c < P; c++) if (logits[o + c] > logits[o + best]) best = c;   // residue logits only, as in extract_neuron_data.py
      if (best === (a + b) % P) ok++;
    }
    return ok / N;
  }

  // neuron-major activations: 512 × Float32Array(1849)
  function transposeActs(hid) {
    const out = new Array(DM);
    for (let j = 0; j < DM; j++) { const r = new Float32Array(N); for (let i = 0; i < N; i++) r[i] = hid[i * DM + j]; out[j] = r; }
    return out;
  }
  function rows(arr, n, w) { const out = new Array(n); for (let j = 0; j < n; j++) out[j] = arr.slice(j * w, (j + 1) * w); return out; }

  // the run's dominant frequencies; set from the run's index.json via setK
  let K = [1, 9, 14], FREQ_MODES = {};
  function setK(k) {
    K = k.slice(); FREQ_MODES = {};
    for (const q of K) FREQ_MODES['freq' + q] = [q, P - q];
    FREQ_MODES.freqall = K.flatMap(q => [q, P - q]);
  }
  setK(K);

  // Data for neuron_viewer / logit_viewer for one mode
  function neuronMode(sd, mode) {
    const WE = mode === 'original' ? null : ablateWE(sd.W_E, FREQ_MODES[mode]);
    const f = forward(sd, WE);
    return { activations: transposeActs(f.hid), accuracy: Math.round(accuracy(f.logits) * 1000) / 10,
             weights: rows(f.WoU, DM, V), weights_pre: rows(sd.W_o, DM, D) };
  }

  // Attention stage for attention_viewer: per-head outputs and the residual stream at query position `pos`
  // (0 = token a, 1 = token b, 2 = the answer slot) for all (a, b). No causal mask: every query sees all 3 keys.
  function attention(sd, pos) {
    const WE = sd.W_E, Wp = sd.W_p, WQ = sd.W_Q, WK = sd.W_K, WV = sd.W_V, WO = sd.W_O;
    const x0 = (p, t) => { const v = new Float32Array(D); for (let d = 0; d < D; d++) v[d] = WE[t * D + d] + Wp[p * D + d]; return v; };
    const slots = [[0, P], [1, P], [2, 1]];                 // (position, number of possible tokens)
    const Q = [], Kk = [], VO = [];
    for (let h = 0; h < H; h++) {
      Q.push([[], [], []]); Kk.push([[], [], []]); VO.push([[], [], []]);
      for (const [p, cnt] of slots) for (let ti = 0; ti < cnt; ti++) {
        const t = p === 2 ? P : ti, x = x0(p, t);
        const q = new Float32Array(DH), k = new Float32Array(DH), v = new Float32Array(DH);
        for (let d = 0; d < D; d++) { const xv = x[d]; if (!xv) continue; const o = h * D * DH + d * DH; for (let e = 0; e < DH; e++) { q[e] += xv * WQ[o + e]; k[e] += xv * WK[o + e]; v[e] += xv * WV[o + e]; } }
        const vo = new Float32Array(D);
        for (let e = 0; e < DH; e++) { const ve = v[e]; if (!ve) continue; const o = h * DH * D + e * D; for (let d = 0; d < D; d++) vo[d] += ve * WO[o + d]; }
        Q[h][p][t] = q; Kk[h][p][t] = k; VO[h][p][t] = vo;
      }
    }
    const dot = (u, w) => { let s = 0; for (let e = 0; e < DH; e++) s += u[e] * w[e]; return s; };
    const heads = Array.from({ length: H }, () => new Float32Array(N * D));
    const stream = new Float32Array(N * D), attn = new Float32Array(N * H * 3);
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const i = a * P + b, row = i * D, tok = [a, b, P], xq = x0(pos, tok[pos]);
      for (let d = 0; d < D; d++) stream[row + d] = xq[d];
      for (let h = 0; h < H; h++) {
        const q = Q[h][pos][tok[pos]];
        const s = [dot(q, Kk[h][0][a]), dot(q, Kk[h][1][b]), dot(q, Kk[h][2][P])];
        const m = Math.max(s[0], s[1], s[2]), e = s.map(v => Math.exp(v - m)), z = e[0] + e[1] + e[2];
        const A = e.map(v => v / z), vs = [VO[h][0][a], VO[h][1][b], VO[h][2][P]], out = heads[h];
        for (let p = 0; p < 3; p++) attn[(i * H + h) * 3 + p] = A[p];
        for (let d = 0; d < D; d++) { const v = A[0] * vs[0][d] + A[1] * vs[1][d] + A[2] * vs[2][d]; out[row + d] = v; stream[row + d] += v; }
      }
    }
    return { heads, stream, attn, pos };
  }

  // Whole-MLP data for mlp_viewer: stream in (x1), pre/post-ReLU hidden, MLP output (hid W_o + b_o, before W_U),
  // the two weight matrices, and per unit the std over inputs and the dominant 2-D frequency of its 43x43 grid.
  function unitStats(src, width, count) {
    const std = new Float32Array(count), dom = new Int16Array(count), domMag = new Float32Array(count);
    const r1 = new Float64Array(P * P), i1 = new Float64Array(P * P), g = new Float64Array(P * P);
    const half = Math.floor(P / 2);
    for (let u = 0; u < count; u++) {
      let s = 0, s2 = 0;
      for (let i = 0; i < N; i++) { const v = src[i * width + u]; g[i] = v; s += v; s2 += v * v; }
      const mean = s / N; std[u] = Math.sqrt(Math.max(s2 / N - mean * mean, 0));
      if (std[u] === 0) { dom[u] = 0; continue; }
      // separable 2-D DFT magnitude; search the peak with (k1, k2) != (0, 0) and report its frequency (k1 or k2, folded to 1..21)
      for (let a = 0; a < P; a++) for (let k2 = 0; k2 < P; k2++) { let sr = 0, si = 0; for (let b = 0; b < P; b++) { const v = g[a * P + b]; sr += v * TW_C[k2 * P + b]; si += v * TW_S[k2 * P + b]; } r1[a * P + k2] = sr; i1[a * P + k2] = si; }
      let best = 0, bk1 = 0, bk2 = 0;
      for (let k1 = 0; k1 < P; k1++) for (let k2 = 0; k2 < P; k2++) {
        if (k1 === 0 && k2 === 0) continue;
        let sr = 0, si = 0;
        for (let a = 0; a < P; a++) { const c = TW_C[k1 * P + a], sn = TW_S[k1 * P + a], xr = r1[a * P + k2], xi = i1[a * P + k2]; sr += xr * c - xi * sn; si += xr * sn + xi * c; }
        const m = sr * sr + si * si; if (m > best) { best = m; bk1 = k1; bk2 = k2; }
      }
      const fold = k => k > half ? P - k : k;
      dom[u] = bk1 ? fold(bk1) : fold(bk2); domMag[u] = Math.sqrt(best);
    }
    return { std, dom, domMag };
  }
  function mlp(sd) {
    const f = forward(sd);
    const out = new Float32Array(N * D), W_o = sd.W_o, b_o = sd.b_o;
    for (let i = 0; i < N; i++) {
      const o = i * D, hrow = i * DM;
      for (let d = 0; d < D; d++) out[o + d] = b_o[d];
      for (let j = 0; j < DM; j++) { const hv = f.hid[hrow + j]; if (!hv) continue; const oo = j * D; for (let d = 0; d < D; d++) out[o + d] += hv * W_o[oo + d]; }
    }
    return { x1: f.x1, pre: f.pre, hid: f.hid, out, W_i: sd.W_i, W_o: sd.W_o, b_i: sd.b_i, b_o: sd.b_o,
             statsIn: unitStats(f.x1, D, D), statsHid: unitStats(f.hid, DM, DM), statsOut: unitStats(out, D, D),
             accuracy: Math.round(accuracy(f.logits) * 1000) / 10 };
  }

  // Raw embedding / unembedding for embed_viewer: both as P x D (W_U transposed), placeholder token dropped.
  function embed(sd) {
    const WE = new Float32Array(P * D), WU = new Float32Array(P * D);
    for (let a = 0; a < P; a++) for (let d = 0; d < D; d++) { WE[a * D + d] = sd.W_E[a * D + d]; WU[a * D + d] = sd.W_U[d * V + a]; }
    return { WE, WU };
  }

  // ---- C^n coordinates (see extract_c3_data.py for the conventions; n = K.length) ----
  function c3(sd, summary) {
    const f = forward(sd);
    const nK = K.length;
    const TH = K.map(k => 2 * Math.PI * k / P);
    const zeros = () => Array.from({ length: nK }, () => [0, 0]);
    // 2-D DFT entries needed per neuron, with twiddles e^{-2πi(ka a + kb b)/P} over the 1849 grid
    const entries = [], E = {};
    const add = (ka, kb) => { const key = ka + ',' + kb; if (E[key] === undefined) { E[key] = entries.length; entries.push([ka, kb]); } return E[key]; };
    add(0, 0);
    for (const k of K) { add(-k, 0); add(0, -k); add(-k, -k); add(-k, k); }
    const EC = entries.map(() => new Float64Array(N)), ES = entries.map(() => new Float64Array(N));
    entries.forEach(([ka, kb], e) => { for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) { const t = -2 * Math.PI * (ka * a + kb * b) / P; EC[e][a * P + b] = Math.cos(t); ES[e][a * P + b] = Math.sin(t); } });
    const s2 = 2 / (P * P);
    const dft2At = (col, stride, e) => { let sr = 0, si = 0; const C = EC[e], S = ES[e]; for (let i = 0; i < N; i++) { const v = col[i * stride]; if (v) { sr += v * C[i]; si += v * S[i]; } } return [sr, si]; };
    const cm = (p, q) => [p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]];
    const ab2 = z => z[0] * z[0] + z[1] * z[1];
    const neurons = new Array(DM); let normMax = 0;
    const kappaSum = zeros(), gammaSum = zeros(), dwSum = zeros();
    for (let j = 0; j < DM; j++) {
      const preCol = f.pre.subarray(j), hCol = f.hid.subarray(j);
      let preE = 0, hE = 0;
      for (let i = 0; i < N; i++) { const p = preCol[i * DM], hv = hCol[i * DM]; preE += p * p; hE += hv * hv; }
      const norm = Math.sqrt(hE);
      const preDC = dft2At(preCol, DM, 0), hDC = dft2At(hCol, DM, 0);
      const preVar = preE * N - ab2(preDC), hVar = hE * N - ab2(hDC);     // Parseval: Σ|F|² = N Σ|f|²
      const a = [], b = [], h = [], d = [], w = [], kappa = [], gamma = [];
      let preAxes = 0, hSum = 0, hDif = 0, hAxes = 0;
      for (let ki = 0; ki < nK; ki++) {
        const k = K[ki];
        const A = dft2At(preCol, DM, E[(-k) + ',0']), B = dft2At(preCol, DM, E['0,' + (-k)]);
        const Hs = dft2At(hCol, DM, E[(-k) + ',' + (-k)]), Hd = dft2At(hCol, DM, E[(-k) + ',' + k]);
        const Ha = dft2At(hCol, DM, E[(-k) + ',0']), Hb = dft2At(hCol, DM, E['0,' + (-k)]);
        preAxes += 2 * (ab2(A) + ab2(B)); hSum += 2 * ab2(Hs); hDif += 2 * ab2(Hd); hAxes += 2 * (ab2(Ha) + ab2(Hb));
        a.push([A[0] * s2, A[1] * s2]); b.push([B[0] * s2, B[1] * s2]);
        h.push([Hs[0] * s2, Hs[1] * s2]); d.push([Hd[0] * s2, Hd[1] * s2]);
        let wr = 0, wi = 0; for (let c = 0; c < P; c++) { const v = f.WoU[j * V + c]; wr += v * TW_C[k * P + c]; wi += v * TW_S[k * P + c]; }
        w.push([wr * 2 / P, wi * 2 / P]);
        const kp = cm(h[ki], w[ki]), gm = cm(h[ki], [w[ki][0], -w[ki][1]]);
        kappa.push([kp[0] / 2, kp[1] / 2]); gamma.push([gm[0] / 2, gm[1] / 2]);
      }
      let wTot = 0, wK = 0;
      for (let k = 1; k < P; k++) { let wr = 0, wi = 0; for (let c = 0; c < P; c++) { const v = f.WoU[j * V + c]; wr += v * TW_C[k * P + c]; wi += v * TW_S[k * P + c]; } const e = wr * wr + wi * wi; wTot += e; if (K.includes(k) || K.includes(P - k)) wK += e; }
      const re = a.map((z, i) => ab2(z) + ab2(b[i]));
      let domIn = 0, domOut = 0; for (let i = 1; i < nK; i++) { if (re[i] > re[domIn]) domIn = i; if (ab2(w[i]) > ab2(w[domOut])) domOut = i; }
      const reSum = re.reduce((p, q) => p + q, 0);
      neurons[j] = { a, b, h, d, w, kappa, gamma, bias: preDC[0] / N, hdc: hDC[0] / N, norm,
        domIn, domOut, purity: reSum > 0 ? re[domIn] / reSum : 0,
        fracPre: preVar > 0 ? preAxes / preVar : 0, fracHsum: hVar > 0 ? hSum / hVar : 0, fracHdif: hVar > 0 ? hDif / hVar : 0,
        fracHaxes: hVar > 0 ? hAxes / hVar : 0, fracW: wTot > 0 ? wK / wTot : 0 };
      if (norm > normMax) normMax = norm;
      for (let ki = 0; ki < nK; ki++) {
        kappaSum[ki][0] += kappa[ki][0]; kappaSum[ki][1] += kappa[ki][1];
        gammaSum[ki][0] += gamma[ki][0]; gammaSum[ki][1] += gamma[ki][1];
        const dw = cm(d[ki], w[ki]); dwSum[ki][0] += dw[0] / 2; dwSum[ki][1] += dw[1] / 2;
      }
    }
    const kappaCross = zeros();
    let alive = 0; const domInCounts = new Array(nK).fill(0);
    for (let j = 0; j < DM; j++) {
      const n = neurons[j]; const isAlive = n.norm > 0.05 * normMax;
      if (isAlive) { alive++; domInCounts[n.domIn]++; }
      for (let ki = 0; ki < nK; ki++) if (n.domIn !== ki) { kappaCross[ki][0] += n.kappa[ki][0]; kappaCross[ki][1] += n.kappa[ki][1]; }
    }
    // per-input output coordinates: L_k = (2/P) Σ_c logits(c) e^{-iθc}; Λ_k = mean_i L_k(i)·conj(x_k(a+b))
    const Lhat = new Array(N), logits = new Array(N);
    const Lambda = zeros();
    let logitDC = 0;
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
      const i = a * P + b, o = i * V, t = (a + b) % P;
      const lr = [];
      for (let ki = 0; ki < nK; ki++) {
        const k = K[ki]; let l0 = 0, l1 = 0;
        for (let c = 0; c < P; c++) { const cc = TW_C[k * P + c], ss = TW_S[k * P + c]; l0 += f.logits[o + c] * cc; l1 += f.logits[o + c] * ss; }
        lr.push([l0 * 2 / P, l1 * 2 / P]);
        const cr = Math.cos(TH[ki] * t), ci = Math.sin(TH[ki] * t);     // conj(x_k(a+b)) = e^{+iθ t}
        Lambda[ki][0] += (lr[ki][0] * cr - lr[ki][1] * ci) / N; Lambda[ki][1] += (lr[ki][0] * ci + lr[ki][1] * cr) / N;
      }
      Lhat[i] = lr;
      const row = f.logits.slice(o, o + P); logits[i] = row;
      for (let c = 0; c < P; c++) logitDC += row[c] / (N * P);
    }
    const biasOut = [];
    for (let ki = 0; ki < nK; ki++) { const k = K[ki]; let r = 0, im = 0; for (let c = 0; c < P; c++) { r += f.bU[c] * TW_C[k * P + c]; im += f.bU[c] * TW_S[k * P + c]; } biasOut.push([r * 2 / P, im * 2 / P]); }
    const weE = new Float64Array(P); let weTot = 0;
    for (let d = 0; d < D; d++) for (let k = 0; k < P; k++) { let sr = 0, si = 0; for (let a = 0; a < P; a++) { const v = sd.W_E[a * D + d]; sr += v * TW_C[k * P + a]; si += v * TW_S[k * P + a]; } weE[k] += sr * sr + si * si; weTot += sr * sr + si * si; }
    const fracWE = { dc: weE[0] / weTot }; for (const k of K) fracWE[k] = (weE[k] + weE[P - k]) / weTot;
    let rmsL = 0;
    for (let i = 0; i < N; i++) { const o = i * V; let mm = 0; for (let c = 0; c < P; c++) mm += f.logits[o + c]; mm /= P; for (let c = 0; c < P; c++) rmsL += (f.logits[o + c] - mm) ** 2; }
    rmsL = Math.sqrt(rmsL / (N * P));
    const attnMean = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < N; i++) for (let h = 0; h < H; h++) for (let p = 0; p < 3; p++) attnMean[h][p] += f.attn[(i * H + h) * 3 + p] / N;
    return {
      modulus: P, K: K.slice(), d_mlp: DM, d_model: D, neurons, acts: transposeActs(f.hid), Lhat, logits,
      totals: { sumKappa: kappaSum, sumGamma: gammaSum, sumKappaCross: kappaCross, sumDw: dwSum, Lambda3d: Lambda, LambdaEmp: Lambda, biasOut, logitDC, alive, domInCounts },
      stream: Object.assign({ fracWE, rmsLogits: rmsL, attnMean, accuracy: accuracy(f.logits) }, summary || {}),
    };
  }

  return { P, D, H, DH, DM, V, N, decode, forward, ablateWE, accuracy, transposeActs, neuronMode, embed, attention, mlp, c3, setK, get K() { return K; }, get FREQ_MODES() { return FREQ_MODES; } };
})();
if (typeof module !== 'undefined') module.exports = ModelJS;
