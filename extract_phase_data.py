"""Extract phase-space data for the 3 main frequencies.

For each frequency k in {1, 9, 14}:
  - O_k(a,b) = Σ_j act_j(a,b) × ŵ_j(k):  the output at freq k for each input pair
  - Per neuron j: ŵ_j(k) complex, peakVal, dominantFreq
  - k_inv: k^{-1} mod P (for reindexing axes by phase)
"""

import torch, json, sys, numpy as np
sys.path.insert(0, '.')
from config import config
from model import model
from encoding import encode_values
from embedding import embedding
from attention import attention

P = 43
c = config(modulus=P)
m = model(c)
m.load_state_dict(torch.load('grokkedweights-mod43.pth', map_location='cpu'))
m.training = False

inputs = []
for a in range(P):
    for b in range(P):
        inputs.append(encode_values(c, a, b, c.d_vocab - 1))
x = torch.stack(inputs)

with torch.no_grad():
    x0 = embedding(m, x)
    x1 = attention(m, x0)
    hidden = torch.relu(x1[:, 2, :] @ m.W_i + m.b_i)  # (P*P, d_mlp)

weights = (m.W_o @ m.W_U).detach().numpy()[:, :P]  # (d_mlp, P)
acts = hidden.numpy()  # (P*P, d_mlp)
d_mlp = acts.shape[1]

# DFT of each neuron's weight vector
W_hat = np.fft.fft(weights, axis=1)  # (d_mlp, P) complex

# Peak activations per neuron
peak_vals = acts.max(axis=0)  # (d_mlp,)

# Classify neurons by dominant output freq
dom_freq = []
for j in range(d_mlp):
    best_k, best_mag = 1, 0
    for k in [1, 9, 14]:
        m2 = abs(W_hat[j, k])
        if m2 > best_mag:
            best_mag = m2
            best_k = k
    dom_freq.append(best_k)

data = { 'modulus': P, 'd_mlp': d_mlp, 'freqs': {} }

for k in [1, 9, 14]:
    k_inv = pow(k, -1, P)  # k^{-1} mod P
    k_conj = P - k          # conjugate frequency

    # O_k(a,b) = Σ_j act_j(a,b) × ŵ_j(k)
    # acts is (P*P, d_mlp), W_hat[:,k] is (d_mlp,) complex
    O_k = acts @ W_hat[:, k]  # (P*P,) complex

    # Per-neuron data for this freq
    neurons = []
    for j in range(d_mlp):
        w_re = float(W_hat[j, k].real)
        w_im = float(W_hat[j, k].imag)
        w_mag = float(abs(W_hat[j, k]))
        neurons.append({
            're': round(w_re, 4),
            'im': round(w_im, 4),
            'mag': round(w_mag, 4),
            'peak': round(float(peak_vals[j]), 4),
            'dom': dom_freq[j],
        })

    data['freqs'][str(k)] = {
        'k': k,
        'k_inv': k_inv,
        'k_conj': k_conj,
        'O_re': [round(float(v), 4) for v in O_k.real],
        'O_im': [round(float(v), 4) for v in O_k.imag],
        'neurons': neurons,
    }

# Also store the raw activations (needed for per-target neuron breakdown)
# Compress: only store activations for neurons with non-trivial contribution
# to any of the 3 freqs (mag > 1% of max)
max_mag = max(abs(W_hat[j, k]) * peak_vals[j]
              for j in range(d_mlp) for k in [1, 9, 14])
active = []
for j in range(d_mlp):
    contrib = max(abs(W_hat[j, k]) * peak_vals[j] for k in [1, 9, 14])
    if contrib > max_mag * 0.001:
        active.append(j)

print(f"Active neurons: {len(active)}/{d_mlp}")
data['active_neurons'] = active
data['activations'] = {}
for j in active:
    data['activations'][str(j)] = [round(float(v), 4) for v in acts[:, j]]

with open('phase_data.json', 'w') as f:
    json.dump(data, f)

sz = len(json.dumps(data)) / 1e6
print(f"Wrote phase_data.json ({sz:.1f} MB)")

# Quick check: verify O_k matches expected phases
for k in [1, 9, 14]:
    fdata = data['freqs'][str(k)]
    O = np.array(fdata['O_re']) + 1j * np.array(fdata['O_im'])
    O = O.reshape(P, P)
    # For a+b=0 mod P, ideal phase = 0
    # For a+b=1 mod P, ideal phase = 2πk/P
    phases = np.angle(O)
    print(f"\nFreq k={k}: phase at (0,0) = {np.degrees(phases[0,0]):.1f}°, "
          f"ideal = {360*k*0/P:.1f}°")
    print(f"  phase at (1,0) = {np.degrees(phases[1,0]):.1f}°, "
          f"ideal = {360*k*1/P:.1f}°")
    print(f"  phase at (0,1) = {np.degrees(phases[0,1]):.1f}°, "
          f"ideal = {360*k*1/P:.1f}°")
    print(f"  |O| range: [{np.abs(O).min():.1f}, {np.abs(O).max():.1f}]")
