"""Extract the frequency-domain input→output tensor C(k1, k2, k_out).

C(k1, k2, k_out) = Σ_j  ĥ_j(k1, k2) × ŵ_j(k_out)

where ĥ_j is the 2D DFT of neuron j's activation grid (43×43),
and ŵ_j is the 1D DFT of its weight vector w_j(c) = W_o[j,:] @ W_U[:,c].

This is equivalently the 3D DFT of T(a, b, c) = Σ_j act_j(a,b) × w_j(c),
the full MLP input→output map.

Output: freq_tensor.json with re[k_out][k1*P+k2] and im[k_out][k1*P+k2]
for k_out = 0..P-1, k1 = 0..P-1, k2 = 0..P-1.
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

# Build all (a, b) inputs
inputs = []
for a in range(P):
    for b in range(P):
        inputs.append(encode_values(c, a, b, c.d_vocab - 1))
x = torch.stack(inputs)

with torch.no_grad():
    x0 = embedding(m, x)
    x1 = attention(m, x0)
    hidden = torch.relu(x1[:, 2, :] @ m.W_i + m.b_i)  # (P*P, d_mlp)

weights = (m.W_o @ m.W_U).detach().numpy()  # (d_mlp, d_vocab=44)
acts = hidden.numpy()  # (P*P, d_mlp)

# Build T(a, b, c) = Σ_j act_j(a,b) × w_j(c)  for c=0..P-1
# acts is (P*P, d_mlp), weights[:, :P] is (d_mlp, P)
# T = acts @ weights[:, :P]  shape (P*P, P) = (1849, 43)
T = acts @ weights[:, :P]  # (P*P, P)
T = T.reshape(P, P, P)     # T[a, b, c]

print(f"T(a,b,c) shape: {T.shape}")
print(f"T range: [{T.min():.2f}, {T.max():.2f}]")

# 3D DFT
C = np.fft.fftn(T)  # C[k1, k2, k_out]

print(f"C(k1,k2,k_out) shape: {C.shape}")

# Rearrange to C[k_out][k1*P+k2] for the viewer
# Store re and im separately, rounded
data = {
    'modulus': P,
    're': [],
    'im': [],
}

for k_out in range(P):
    plane = C[:, :, k_out]  # (P, P) complex
    re_flat = plane.real.flatten().tolist()
    im_flat = plane.imag.flatten().tolist()
    data['re'].append([round(v, 4) for v in re_flat])
    data['im'].append([round(v, 4) for v in im_flat])

with open('freq_tensor.json', 'w') as f:
    json.dump(data, f)

sz = len(json.dumps(data)) / 1e6
print(f"Wrote freq_tensor.json ({sz:.1f} MB)")

# Quick analysis: where is the energy?
mag = np.abs(C)
print(f"\nTop 20 entries by magnitude:")
flat = mag.flatten()
top_idx = np.argsort(flat)[::-1][:20]
for idx in top_idx:
    k1, k2, k_out = np.unravel_index(idx, (P, P, P))
    print(f"  C({k1}, {k2}, {k_out}) = {C[k1,k2,k_out]:.2f}  |{mag[k1,k2,k_out]:.2f}|")
