"""Project the grokked mod-43 model onto C^3 (frequencies 1, 9, 14).

Conventions (P = 43, theta_k = 2*pi*k/P):
  token a  ->  x_k(a) = exp(-i theta_k a)             (its DFT coefficient at k)

Per neuron j, with pre_j(a,b) the pre-activation grid and h_j = ReLU(pre_j):
  pre_j(a,b) = bias_j + sum_k Re( a_jk x_k(a) + b_jk x_k(b) ) + (tiny rest)
  h_j(a,b)  contains  Re( h_jk x_k(a+b) )  with  h_jk ~ rho * a_jk * b_jk, rho > 0
             and      Re( d_jk x_k(a-b) )  (difference term, cancels across neurons)
  w_j(c) = (W_o W_U)[j, c]  contains  Re( w_jk e^{+i theta_k c} )

So neuron j contributes  h_j(a,b) * w_jk  to the output coordinate (the MLP output is the
whole stream before unembedding: mlp.py has no skip connection)
  L_k(a,b) := (2/P) DFT_c(logits(a,b,.))[k]
and, in the x_k(a+b) component, the coefficient kappa_jk = h_jk * w_jk / 2.
Correct behaviour: sum_j kappa_jk real positive, so L_k ~ Lambda_k x_k(a) x_k(b).

Writes c3_data.json for c3_viewer.html.
"""

import json, sys
import numpy as np
import torch

sys.path.insert(0, '.')
from config import config
from model import model
from encoding import encode_values
from embedding import embedding
from attention import attention

P = 43
K = [1, 9, 14]
c = config(modulus=P)
m = model(c)
m.load_state_dict(torch.load('grokkedweights-mod43.pth', map_location='cpu'))

x = torch.stack([encode_values(c, a, b, P) for a in range(P) for b in range(P)])
with torch.no_grad():
    x0 = embedding(m, x)
    x1 = attention(m, x0)
    S = x0.unsqueeze(-3) @ m.W_Q.unsqueeze(-4) @ m.W_K.unsqueeze(-4).mT @ x0.unsqueeze(-3).mT
    attn = (S - S.max(-1, keepdim=True).values).softmax(-1)[:, :, 2, :].numpy()  # (N, heads, 3)
    X1 = x1[:, 2, :].numpy()
    W_i, b_i = m.W_i.numpy(), m.b_i.numpy()
    W_o, b_o = m.W_o.numpy(), m.b_o.numpy()
    W_U = m.W_U.numpy()[:, :P]
    W_E = m.W_E.numpy()[:P]
    W_V, W_O = m.W_V.numpy(), m.W_O.numpy()

N = P * P
targets = np.array([(a + b) % P for a in range(P) for b in range(P)])
pre = X1 @ W_i + b_i                      # (N, 512)
h = np.maximum(pre, 0)
mlp_out = h @ W_o + b_o
bias_out = b_o @ W_U                      # (43,)
logits = mlp_out @ W_U                    # mlp.py has no skip connection: the MLP output replaces the stream
w = W_o @ W_U                             # (512, 43)
assert (logits.argmax(1) == targets).all()

# ---------------- per-neuron C^3 coefficients ----------------
Fp = np.fft.fft2(pre.reshape(P, P, -1), axes=(0, 1))   # (P, P, 512)
Fh = np.fft.fft2(h.reshape(P, P, -1), axes=(0, 1))
Fw = np.fft.fft(w, axis=1)                             # (512, P)
s2 = 2.0 / P**2
a_c = np.stack([Fp[-k, 0] for k in K], 1) * s2          # pre ⊃ Re(a x_k(a))
b_c = np.stack([Fp[0, -k] for k in K], 1) * s2
h_c = np.stack([Fh[-k, -k] for k in K], 1) * s2         # h ⊃ Re(h x_k(a+b))
d_c = np.stack([Fh[-k, k] for k in K], 1) * s2          # h ⊃ Re(d x_k(a-b))
w_c = Fw[:, K] * 2.0 / P                                # w(c) ⊃ Re(w e^{+i theta c})
kappa = h_c * w_c / 2                                   # coefficient of x_k(a+b) in L_k
gamma = h_c * np.conj(w_c) / 2                          # junk: coefficient of x_k(a+b+c)
bias_pre = Fp[0, 0].real / P**2
hnorm = np.sqrt((h**2).sum(0))
read_energy = np.abs(a_c)**2 + np.abs(b_c)**2
domIn = read_energy.argmax(1)
domOut = np.abs(w_c).argmax(1)
purity = read_energy.max(1) / np.maximum(read_energy.sum(1), 1e-12)
# fraction of pre-activation variance explained by the 12 axis coefficients at K
pre_var = (np.abs(Fp)**2).sum((0, 1)) - np.abs(Fp[0, 0])**2
pre_axes = sum(np.abs(Fp[k, 0])**2 + np.abs(Fp[-k, 0])**2 + np.abs(Fp[0, k])**2 + np.abs(Fp[0, -k])**2 for k in K)
frac_pre = pre_axes / np.maximum(pre_var, 1e-12)
# fraction of ReLU-output variance in the sum (k,k) and difference (k,-k) terms at K
h_var = (np.abs(Fh)**2).sum((0, 1)) - np.abs(Fh[0, 0])**2
h_sum = sum(np.abs(Fh[k, k])**2 + np.abs(Fh[-k, -k])**2 for k in K)
h_dif = sum(np.abs(Fh[k, -k])**2 + np.abs(Fh[-k, k])**2 for k in K)
h_axes = sum(np.abs(Fh[k, 0])**2 + np.abs(Fh[-k, 0])**2 + np.abs(Fh[0, k])**2 + np.abs(Fh[0, -k])**2 for k in K)
frac_h_sum = h_sum / np.maximum(h_var, 1e-12)
frac_h_dif = h_dif / np.maximum(h_var, 1e-12)
frac_h_axes = h_axes / np.maximum(h_var, 1e-12)
w_var = (np.abs(Fw[:, 1:])**2).sum(1)
frac_w = sum(np.abs(Fw[:, k])**2 + np.abs(Fw[:, -k])**2 for k in K) / np.maximum(w_var, 1e-12)

def cl(z):  # complex -> [re, im] rounded
    return [round(float(z.real), 5), round(float(z.imag), 5)]

neurons = []
for j in range(c.d_mlp):
    neurons.append({
        'a': [cl(z) for z in a_c[j]], 'b': [cl(z) for z in b_c[j]],
        'h': [cl(z) for z in h_c[j]], 'd': [cl(z) for z in d_c[j]],
        'w': [cl(z) for z in w_c[j]],
        'kappa': [cl(z) for z in kappa[j]], 'gamma': [cl(z) for z in gamma[j]],
        'bias': round(float(bias_pre[j]), 4), 'hdc': round(float(Fh[0, 0, j].real / P**2), 4),
        'norm': round(float(hnorm[j]), 3),
        'domIn': int(domIn[j]), 'domOut': int(domOut[j]), 'purity': round(float(purity[j]), 4),
        'fracPre': round(float(frac_pre[j]), 4), 'fracHsum': round(float(frac_h_sum[j]), 4),
        'fracHdif': round(float(frac_h_dif[j]), 4), 'fracHaxes': round(float(frac_h_axes[j]), 4),
        'fracW': round(float(frac_w[j]), 4),
    })

# ---------------- per-input C^3 output coordinates ----------------
Fb = np.fft.fft(bias_out)[K] * 2.0 / P
FL = np.fft.fft(logits, axis=1)[:, K] * 2.0 / P        # L_k(a,b)
x_ab = np.exp(-2j * np.pi * np.outer(targets, K) / P)  # x_k(a+b)
Lambda_emp = (FL * np.conj(x_ab)).mean(0)              # should be real positive
F3 = np.fft.fftn(logits.reshape(P, P, P))
Lambda_3d = np.array([2.0 / P**3 * F3[-k, -k, k] for k in K])   # coefficient of x_k(a+b) in L_k

alive = hnorm > 0.05 * hnorm.max()
totals = {
    'sumKappa': [cl(z) for z in kappa.sum(0)],
    'sumGamma': [cl(z) for z in gamma.sum(0)],
    'sumKappaCross': [cl(kappa[domIn != i, i].sum()) for i in range(3)],
    'sumDw': [cl(z) for z in (d_c * w_c / 2).sum(0)],
    'sumDwBar': [cl(z) for z in (d_c * np.conj(w_c) / 2).sum(0)],
    'Lambda3d': [cl(z) for z in Lambda_3d],
    'LambdaEmp': [cl(z) for z in Lambda_emp],
    'biasOut': [cl(z) for z in Fb],
    'logitDC': round(float(logits.mean()), 4),
    'alive': int(alive.sum()),
    'domInCounts': [int(((domIn == i) & alive).sum()) for i in range(3)],
}

# ---------------- stream-dimension checks ----------------
targets_t = targets
def acc_from(X1v, out_fn=None):
    hh = np.maximum(X1v @ W_i + b_i, 0)
    X2 = hh @ W_o + b_o
    if out_fn is not None:
        X2 = out_fn(X2)
    return float((np.argmax(X2 @ W_U, -1) == targets_t).mean())

F_E = np.fft.fft(W_E, axis=0)
E_tot = (np.abs(F_E)**2).sum()
S6 = np.array(sum([[F_E[k].real, F_E[k].imag] for k in K], []))      # embedding freq vectors (6,128)
Q6, _ = np.linalg.qr(S6.T)
F_U = np.fft.fft(W_U, axis=1)
S6o = np.array(sum([[F_U[:, k].real, F_U[:, k].imag] for k in K], []))
Q6o, _ = np.linalg.qr(S6o.T)
OVa = sum(W_V[hd] @ W_O[hd] for hd in (0, 1))
OVb = sum(W_V[hd] @ W_O[hd] for hd in (2, 3))
Q12, _ = np.linalg.qr(np.concatenate([S6 @ OVa, S6 @ OVb]).T)
X1m = X1.mean(0); V = X1 - X1m
Vo = mlp_out - mlp_out.mean(0)
sv_E = np.linalg.svd(W_E - W_E.mean(0), compute_uv=False)
sv_X1 = np.linalg.svd(V, compute_uv=False)
sv_mo = np.linalg.svd(Vo, compute_uv=False)
G = np.fft.fft2(V.reshape(P, P, -1), axes=(0, 1))
Gtot = (np.abs(G)**2).sum()
rng = np.random.default_rng(0)
R6, _ = np.linalg.qr(rng.standard_normal((128, 6)))
R12, _ = np.linalg.qr(rng.standard_normal((128, 12)))
def out6(X2):
    mu = X2.mean(0); return mu + (X2 - mu) @ Q6o @ Q6o.T
F_L = np.fft.fft(logits, axis=1)
tot3 = (np.abs(F3)**2).sum() - abs(F3[0, 0, 0])**2
stream = {
    'svWE': [round(float(v), 4) for v in sv_E],
    'svX1': [round(float(v), 3) for v in sv_X1[:32]],
    'svMLPout': [round(float(v), 3) for v in sv_mo[:32]],
    'fracWE': {'dc': round(float((np.abs(F_E[0])**2).sum() / E_tot), 4),
               **{str(k): round(float((np.abs(F_E[k])**2 + np.abs(F_E[-k])**2).sum() / E_tot), 4) for k in K}},
    'fracX1onAxesK': round(float(sum((np.abs(G[k, 0])**2).sum() + (np.abs(G[-k, 0])**2).sum()
                                     + (np.abs(G[0, k])**2).sum() + (np.abs(G[0, -k])**2).sum() for k in K) / Gtot), 5),
    'fracX1in6embed': round(float(((V @ Q6 @ Q6.T)**2).sum() / (V**2).sum()), 4),
    'fracX1in12': round(float(((V @ Q12 @ Q12.T)**2).sum() / (V**2).sum()), 5),
    'fracMLPoutIn6': round(float(((Vo @ Q6o @ Q6o.T)**2).sum() / (Vo**2).sum()), 4),
    'fracLogitsK': round(float((np.abs(F_L[:, K])**2).sum() * 2 / (np.abs(F_L[:, 1:])**2).sum()), 4),
    'fracLogits3D': round(float(sum(abs(F3[k, k, -k])**2 + abs(F3[-k, -k, k])**2 for k in K) / tot3), 4),
    'rmsLogits': round(float(np.sqrt(((logits - logits.mean(1, keepdims=True))**2).mean())), 3),
    'cosInOut': [round(float(v), 3) for v in np.linalg.svd(Q6.T @ Q6o, compute_uv=False)],
    'attnMean': [[round(float(v), 3) for v in row] for row in attn.mean(0)],
    'attnStd': round(float(attn.std(0).max()), 4),
    'acc': {
        'baseline': acc_from(X1),
        'in6embed': acc_from(X1m + V @ Q6 @ Q6.T),
        'in12': acc_from(X1m + V @ Q12 @ Q12.T),
        'in6pca': acc_from(X1m + V @ np.linalg.svd(V, full_matrices=False)[2][:6].T @ np.linalg.svd(V, full_matrices=False)[2][:6]),
        'out6': acc_from(X1, out6),
        'in12out6': acc_from(X1m + V @ Q12 @ Q12.T, out6),
        'random6': acc_from(X1m + V @ R6 @ R6.T),
        'random12': acc_from(X1m + V @ R12 @ R12.T),
    },
}

data = {
    'modulus': P, 'K': K, 'd_mlp': c.d_mlp, 'd_model': c.d_model,
    'neurons': neurons,
    'acts': [[round(float(v), 3) for v in h[:, j]] for j in range(c.d_mlp)],
    'Lhat': [[cl(z) for z in row] for row in FL],
    'logits': [[round(float(v), 2) for v in row] for row in logits],
    'totals': totals,
    'stream': stream,
}
with open('c3_data.json', 'w') as f:
    json.dump(data, f, separators=(',', ':'))

print('alive neurons:', totals['alive'], ' dominant-freq counts:', dict(zip(K, totals['domInCounts'])))
for i, k in enumerate(K):
    print(f'k={k:2d}: Lambda (3D DFT) = {abs(Lambda_3d[i]):.2f} at {np.degrees(np.angle(Lambda_3d[i])):.1f} deg ; '
          f'sum kappa = {abs(kappa[:, i].sum()):.2f} at {np.degrees(np.angle(kappa[:, i].sum())):.1f} deg ; '
          f'|sum gamma| = {abs(gamma[:, i].sum()):.2f}')
print('accuracies:', stream['acc'])
import os
print(f'wrote c3_data.json ({os.path.getsize("c3_data.json") / 1e6:.1f} MB)')
