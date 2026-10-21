"""Export a model run for the viewers: a directory of training snapshots, or a single final model.

  python export_snapshots.py --weights weights --out snapshots --name "training run"
  python export_snapshots.py --final grokkedweights-mod43.pth --out snapshots_final --name "final grokked model"

Writes <out>/<step>.bin  — all parameters as little-endian float16 in a fixed order
       <out>/index.json — steps, the run's dominant frequencies K (auto-detected from W_E at the last
                          step unless --K is given), train split, and a per-step summary (train/test loss
                          and accuracy, W_E frequency energies, Lambda_k, stream singular values and
                          projection accuracies), and updates runs.json at the repo root.
The viewers run the forward pass themselves (model_js.js), so no activations are stored.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '.')
from config import config
from model import model
from encoding import encode_values
from embedding import embedding
from attention import attention

ap = argparse.ArgumentParser()
ap.add_argument('--weights', help='directory of <step>.pt snapshots')
ap.add_argument('--final', help='a single state-dict file (one "step")')
ap.add_argument('--out', required=True)
ap.add_argument('--name', default=None)
ap.add_argument('--K', default=None, help='comma-separated frequencies, e.g. 1,9,14 (default: auto)')
ap.add_argument('--train', default='in_training_mod43.pt', help='one-hot training inputs defining the train/test split (ignored with --final)')
ap.add_argument('--min_energy', type=float, default=0.05, help='auto-K: keep frequencies with ≥ this share of non-DC W_E energy')
args = ap.parse_args()

P = 43
ORDER = ['W_E', 'W_p', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_i', 'b_i', 'W_o', 'b_o', 'W_U']
c = config(modulus=P)
os.makedirs(args.out, exist_ok=True)

if args.final:
    files, steps = [args.final], [0]
else:
    files = sorted(glob.glob(os.path.join(args.weights, '*.pt')), key=lambda f: int(os.path.basename(f)[:-3]))
    steps = [int(os.path.basename(f)[:-3]) for f in files]
assert files, 'no snapshots found'

# --- dominant frequencies of the run, from W_E at the last step ---
last_sd = torch.load(files[-1], map_location='cpu')
F_last = np.fft.fft(last_sd['W_E'].numpy()[:P], axis=0)
e = (np.abs(F_last) ** 2).sum(1); nd = e[1:P // 2 + 1] + e[P // 2 + 1:][::-1]; nd = nd / nd.sum()
if args.K:
    K = [int(k) for k in args.K.split(',')]
else:
    K = [k + 1 for k in range(P // 2) if nd[k] >= args.min_energy]
nK = len(K)
print('frequencies K =', K, ' (W_E energy shares at last step: ' + ', '.join(f'k={k}: {nd[k-1]*100:.0f}%' for k in K) + ')')

x_all = torch.stack([encode_values(c, a, b, P) for a in range(P) for b in range(P)])
targets = np.array([(a + b) % P for a in range(P) for b in range(P)])
if args.final or not os.path.exists(args.train):
    train_pairs = []
    train_mask = np.ones(P * P, bool)     # no split known: "train" = everything
else:
    tr_in = torch.load(args.train)
    train_pairs = sorted(set(zip(tr_in[:, 0].argmax(-1).tolist(), tr_in[:, 1].argmax(-1).tolist())))
    train_mask = np.zeros(P * P, bool)
    for a, b in train_pairs: train_mask[a * P + b] = True
test_mask = ~train_mask if train_pairs else train_mask

m = model(c)
summ = []
t0 = time.time()
for n, (f, step) in enumerate(zip(files, steps)):
    sd = torch.load(f, map_location='cpu')
    with open(os.path.join(args.out, f'{step}.bin'), 'wb') as out:
        for k in ORDER:
            out.write(sd[k].detach().numpy().astype('<f2').tobytes())
    m.load_state_dict(sd)
    with torch.no_grad():
        x0 = embedding(m, x_all); x1 = attention(m, x0)
        X1 = x1[:, 2, :].numpy()
        Satt = x0.unsqueeze(-3) @ m.W_Q.unsqueeze(-4) @ m.W_K.unsqueeze(-4).mT @ x0.unsqueeze(-3).mT
        attn = (Satt - Satt.max(-1, keepdim=True).values).softmax(-1)[:, :, 2, :].mean(0).numpy()
    W_i, b_i, W_o, b_o = (sd[k].numpy() for k in ('W_i', 'b_i', 'W_o', 'b_o'))
    W_U = sd['W_U'].numpy(); W_E = sd['W_E'].numpy()[:P]
    W_V, W_O = sd['W_V'].numpy(), sd['W_O'].numpy()
    pre = X1 @ W_i + b_i; h = np.maximum(pre, 0); mo = h @ W_o + b_o
    logits_all = mo @ W_U                               # (1849, 44); mlp.py has no skip connection
    mx = logits_all.max(1, keepdims=True)
    lsm = logits_all - mx - np.log(np.exp(logits_all - mx).sum(1, keepdims=True))
    nll = -lsm[np.arange(P * P), targets]
    pred = logits_all.argmax(1)
    rec = {
        'step': step,
        'train_loss': float(nll[train_mask].mean()), 'test_loss': float(nll[test_mask].mean()),
        'train_acc': float((pred == targets)[train_mask].mean()), 'test_acc': float((pred == targets)[test_mask].mean()),
    }
    F_E = np.fft.fft(W_E, axis=0); e = (np.abs(F_E) ** 2).sum(1); tot = e.sum()
    rec['we'] = [float(e[0] / tot)] + [float((e[k] + e[P - k]) / tot) for k in range(1, P // 2 + 1)]
    L = logits_all[:, :P].reshape(P, P, P)
    F3 = np.fft.fftn(L)
    rec['Lambda'] = [[float((2 / P**3 * F3[-k, -k, k]).real), float((2 / P**3 * F3[-k, -k, k]).imag)] for k in K]
    tot3 = (np.abs(F3) ** 2).sum() - abs(F3[0, 0, 0]) ** 2
    rec['frac3d'] = float(sum(abs(F3[k, k, -k]) ** 2 + abs(F3[-k, -k, k]) ** 2 for k in K) / tot3) if tot3 > 0 else 0.0
    X1m = X1.mean(0); V = X1 - X1m; Vo = mo - mo.mean(0)
    rec['svWE'] = [float(v) for v in np.linalg.svd(W_E - W_E.mean(0), compute_uv=False)[:max(10, 2 * nK + 4)]]
    rec['svX1'] = [float(v) for v in np.linalg.svd(V, compute_uv=False)[:max(20, 4 * nK + 8)]]
    rec['svMLP'] = [float(v) for v in np.linalg.svd(Vo, compute_uv=False)[:max(20, 2 * nK + 8)]]
    S_in = np.array(sum([[F_E[k].real, F_E[k].imag] for k in K], []))          # (2nK, 128)
    Q_in, _ = np.linalg.qr(S_in.T)
    OVa = sum(W_V[hd] @ W_O[hd] for hd in (0, 1)); OVb = sum(W_V[hd] @ W_O[hd] for hd in (2, 3))
    Q_ov, _ = np.linalg.qr(np.concatenate([S_in @ OVa, S_in @ OVb]).T)        # (128, 4nK)
    F_U = np.fft.fft(W_U[:, :P], axis=1)
    S_out = np.array(sum([[F_U[:, k].real, F_U[:, k].imag] for k in K], []))
    Q_out, _ = np.linalg.qr(S_out.T)
    def acc_from(X1v, out=False):
        hh = np.maximum(X1v @ W_i + b_i, 0); X2 = hh @ W_o + b_o
        if out:
            mu = X2.mean(0); X2 = mu + (X2 - mu) @ Q_out @ Q_out.T
        return float((np.argmax(X2 @ W_U, -1) == targets).mean())
    rec['fracX1inOV'] = float(((V @ Q_ov @ Q_ov.T) ** 2).sum() / max((V ** 2).sum(), 1e-12))
    rec['fracMLPinOut'] = float(((Vo @ Q_out @ Q_out.T) ** 2).sum() / max((Vo ** 2).sum(), 1e-12))
    rec['accInEmb'] = acc_from(X1m + V @ Q_in @ Q_in.T)
    rec['accInOV'] = acc_from(X1m + V @ Q_ov @ Q_ov.T)
    rec['accOut'] = acc_from(X1, True)
    rec['attn'] = [[float(v) for v in row] for row in attn]
    summ.append(rec)
    if n % 50 == 0:
        print(f'{n}/{len(files)} step {step}: train {rec["train_loss"]:.3f}/{rec["train_acc"]:.2f} test {rec["test_loss"]:.3f}/{rec["test_acc"]:.2f}  ({time.time() - t0:.0f}s)', flush=True)

name = args.name or (os.path.basename(args.final) if args.final else os.path.basename(args.weights))
index = {
    'name': name, 'modulus': P, 'K': K, 'order': ORDER, 'dtype': 'float16',
    'shapes': {k: list(last_sd[k].shape) for k in ORDER},
    'steps': steps, 'single': len(steps) == 1, 'train_pairs': train_pairs,
    'summary': summ,
}
with open(os.path.join(args.out, 'index.json'), 'w') as f:
    json.dump(index, f, separators=(',', ':'))
runs = json.load(open('runs.json')) if os.path.exists('runs.json') else []
runs = [r for r in runs if r['dir'] != args.out] + [{'dir': args.out, 'name': name, 'K': K, 'steps': len(steps)}]
runs.sort(key=lambda r: r['steps'])
json.dump(runs, open('runs.json', 'w'), indent=1)
print('wrote', len(steps), 'snapshot(s) to', args.out, f'({sum(os.path.getsize(os.path.join(args.out, f"{s}.bin")) for s in steps) / 1e6:.0f} MB); runs.json updated')
