"""
Testing loss: ~2.0156302452087402

Example output:

oriuznni
chh
itana
ruhu
tylah
emierush
jabelsentalana
wrajon
nisune
nzili
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# Create the dataset

context_size = 3
X = []; Y = []

for w in words:
    # print(w)
    context = [0] * context_size
    for char in w + '.':
        idx = stoi[char]
        X.append(context)
        Y.append(idx)
        # print(''.join(itos[idx] for idx in context), "--------->", itos[idx])
        context = context[1:] + [idx]

X = torch.tensor(X)
Y = torch.tensor(Y)
g = torch.Generator().manual_seed(2147483647)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g) * 1/(150**0.5)
b1 = torch.randn((200,), generator=g) * 0.01
W2 = torch.randn((200, 27), generator=g) * (5/3)/(150**0.5)
b2 = torch.randn((27,), generator=g) * 0

bn_gain = torch.ones((1, 200))
bn_shift = torch.ones((1, 200))
bn_mean_running = torch.zeros((1, 200))
bn_std_running = torch.ones((1, 200))

parameters = [C, W1, W2, b2, bn_gain, bn_shift] # left out b1

for p in parameters:
    p.requires_grad = True

for epoch in range(10000):
    for p in parameters:
        p.grad = None


    idx = torch.randint(0, X.shape[0], (32, ))

    embedding = C[X[idx]]

    hpreact = embedding.view(-1, 30) @ W1 #+ b1
    bn_mean_iter = hpreact.mean(dim=0, keepdim=False)
    bn_std_iter = hpreact.std(dim=0, keepdim=False)
    hpreact = bn_gain * (hpreact - bn_mean_iter) / (bn_std_iter + 1e-8) + bn_shift

    with torch.no_grad():
        bn_mean_running = 0.999 * bn_mean_running + 0.001 * bn_mean_iter
        bn_std_running = 0.999 * bn_std_running + 0.001 * bn_std_iter

    h = torch.tanh(hpreact)

    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[idx])


    loss.backward()

    for p in parameters:
        p.data = p.data - 0.01 * p.grad


print(loss.item())

""" Not needed if maintaining an exponentially weighted average of running mean + std
with torch.no_grad():
    embedding = C[X]
    hpreact = embedding.view(-1, 30) @ W1 + b1
    bn_mean = hpreact.mean(dim=0, keepdim=False)
    bn_std = hpreact.std(dim=0, keepdim=False)
"""


context_size = 3

for _ in range(10):
    context = [0] * context_size
    out = []
    while True:
        embedding = C[torch.tensor(context).view(1, -1)]
        hpreact = embedding.view(-1, 30) @ W1 + b1
        hpreact = bn_gain * ((hpreact - bn_mean_running) / (bn_std_running + 0.01)) + bn_shift
        h = torch.tanh(hpreact)

        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs[0], num_samples=1, generator=g).item()
        if idx == 0:
            break
        else:
            context = context[1:] + [idx]
            out.append(itos[idx])

    print(''.join(out))
