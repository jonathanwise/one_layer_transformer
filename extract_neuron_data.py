"""Extract MLP neuron activations and weights from the grokked mod-43 model.

Produces neuron_data.json with original and frequency-ablated variants.
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
x = torch.stack(inputs)  # (P*P, 3, d_vocab)

def extract_activations(mdl):
    """Run forward pass through embedding+attention+MLP hidden, return activations."""
    with torch.no_grad():
        x0 = embedding(mdl, x)
        x1 = attention(mdl, x0)
        hidden = torch.relu(x1[:, 2, :] @ mdl.W_i + mdl.b_i)
    return hidden

def check_accuracy(mdl):
    with torch.no_grad():
        from mlp import mlp
        from unembedding import unembedding
        x0 = embedding(mdl, x)
        x1 = attention(mdl, x0)
        x2 = mlp(mdl, x1)
        logits = unembedding(mdl, x2)[:, 2, :]
    preds = logits[:, :P].argmax(dim=-1)
    targets = torch.tensor([(a + b) % P for a in range(P) for b in range(P)])
    return (preds == targets).float().mean().item()

def ablate_embedding(mdl, keep_freqs):
    """Replace W_E with version keeping only specified DFT frequencies.

    keep_freqs: set of frequency indices to keep (e.g. {1, 42} for freq 1).
    Operates on the first P rows (residue tokens). Unknown token row unchanged.
    """
    W_E = mdl.W_E.detach().numpy().copy()
    residues = W_E[:P, :]  # (43, 128)
    F = np.fft.fft(residues, axis=0)
    mask = np.zeros(P)
    for k in keep_freqs:
        mask[k] = 1.0
    F_ablated = F * mask[:, None]
    residues_ablated = np.fft.ifft(F_ablated, axis=0).real
    W_E_new = W_E.copy()
    W_E_new[:P, :] = residues_ablated
    mdl.W_E = torch.nn.Parameter(torch.tensor(W_E_new, dtype=torch.float32))

def to_list(t, decimals=4):
    return [[round(v, decimals) for v in row] for row in t.tolist()]

# --- Original model ---
print("=== Original model ===")
hidden_orig = extract_activations(m)
acc_orig = check_accuracy(m)
print(f"Accuracy: {acc_orig*100:.1f}%")

acts_orig = hidden_orig.reshape(P, P, -1).permute(2, 0, 1)
weights = (m.W_o @ m.W_U)   # (d_mlp, d_vocab) — post-unembedding
weights_pre = m.W_o.detach() # (d_mlp, d_model) — pre-unembedding

# --- Ablated variants: each single frequency and all three ---
freq_configs = [
    ('freq1',  {1, 42},         'freq 1 only (k=1,42)'),
    ('freq9',  {9, 34},         'freq 9 only (k=9,34)'),
    ('freq14', {14, 29},        'freq 14 only (k=14,29)'),
    ('freq3',  {1,42,9,34,14,29}, 'freqs 1,9,14 (k=1,42,9,34,14,29)'),
]

ablated = {}
for key, freqs, label in freq_configs:
    print(f"\n=== Ablated: {label} ===")
    mi = model(c)
    mi.load_state_dict(torch.load('grokkedweights-mod43.pth', map_location='cpu'))
    mi.training = False
    ablate_embedding(mi, freqs)
    hi = extract_activations(mi)
    ai = check_accuracy(mi)
    print(f"Accuracy: {ai*100:.1f}%")
    acts_i = hi.reshape(P, P, -1).permute(2, 0, 1)
    ablated[key] = {
        'activations': to_list(acts_i.reshape(c.d_mlp, P * P)),
        'accuracy': round(ai * 100, 1),
        'label': label,
    }

# --- Write JSON ---
data = {
    'modulus': P,
    'd_mlp': c.d_mlp,
    'd_model': c.d_model,
    'weights': to_list(weights),
    'weights_pre': to_list(weights_pre),
    'original': {
        'activations': to_list(acts_orig.reshape(c.d_mlp, P * P)),
        'accuracy': round(acc_orig * 100, 1),
    },
}
data.update(ablated)

with open('neuron_data.json', 'w') as f:
    json.dump(data, f)

sz = len(json.dumps(data)) / 1e6
print(f"\nWrote neuron_data.json ({sz:.1f} MB)")
