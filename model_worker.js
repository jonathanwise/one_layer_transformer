// model_worker.js — fetches a training snapshot and computes viewer data from its weights.
importScripts('model_js.js?v=5');
let INDEX = null, BASE = 'snapshots';
const cache = new Map();          // step -> decoded state dict (LRU, small)
const MAX_CACHE = 24;
let latestId = 0;

async function getSnapshot(step) {
  if (cache.has(step)) { const sd = cache.get(step); cache.delete(step); cache.set(step, sd); return sd; }
  const r = await fetch(`${BASE}/${step}.bin`);
  if (!r.ok) throw new Error(`snapshot ${step} missing (${r.status})`);
  const sd = ModelJS.decode(await r.arrayBuffer(), INDEX);
  cache.set(step, sd);
  if (cache.size > MAX_CACHE) cache.delete(cache.keys().next().value);
  return sd;
}
const yieldToQueue = () => new Promise(res => setTimeout(res, 0));

self.onmessage = async e => {
  const msg = e.data;
  if (msg.type === 'init') { INDEX = msg.index; BASE = msg.base || 'snapshots'; ModelJS.setK(INDEX.K); return; }
  if (msg.type !== 'compute') return;
  latestId = msg.id;
  let sd;
  try { sd = await getSnapshot(msg.step); }
  catch (err) { self.postMessage({ type: 'error', id: msg.id, step: msg.step, message: String(err) }); return; }
  for (const mode of msg.modes) {
    if (latestId !== msg.id) return;                 // a newer request superseded this one
    let data, transfer = [];
    try {
      if (mode === 'c3') {
        data = ModelJS.c3(sd, msg.summary);
        transfer = data.acts.map(a => a.buffer);
      } else if (mode.startsWith('attn')) {
        data = ModelJS.attention(sd, +mode.slice(4));
        transfer = [...data.heads.map(a => a.buffer), data.stream.buffer, data.attn.buffer];
      } else if (mode === 'mlp') {
        data = ModelJS.mlp(sd);
        transfer = [data.x1.buffer, data.pre.buffer, data.hid.buffer, data.out.buffer];
      } else if (mode === 'embed') {
        data = ModelJS.embed(sd);
        transfer = [data.WE.buffer, data.WU.buffer];
      } else {
        data = ModelJS.neuronMode(sd, mode);
        transfer = data.activations.map(a => a.buffer);
      }
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, step: msg.step, message: `${mode}: ${err && err.message || err}` });
      return;
    }
    self.postMessage({ type: 'result', id: msg.id, step: msg.step, mode, data }, transfer);
    await yieldToQueue();
  }
};
