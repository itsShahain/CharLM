"""
Testing loss: ~2.14

Example output:

mariyah
jailia
krishaira
jana
salas
elaiyanialace
jemmalis
emrey
karlous
elisha
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

C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g) * 1/(150**0.5) # added Kaiming He initialization
b1 = torch.randn((200,), generator=g) * 0.01
W2 = torch.randn((200, 27), generator=g) * (5/3)/(150**0.5)
b2 = torch.randn((27,), generator=g) * 0

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

parameters = [C, W1, b1, W2, b2]

for epoch in range(50000):
    for p in parameters:
        p.grad = None

    idx = torch.randint(0, X.shape[0], (32, ))

    embedding = C[X[idx]]
    h = embedding.view(-1, 150) @ W1 + b1
    logits = torch.tanh(h @ W2 + b2)
    loss = F.cross_entropy(logits, Y[idx])

    loss.backward()

    for p in parameters:
        p.data = p.data - 0.1 * p.grad

print(loss.item())

context_size = 3

for _ in range(10):
    context = [0] * context_size
    out = []
    while True:
        embedding = C[torch.tensor(context).view(1, -1)]
        h = embedding.view(1, -1) @ W1 + b1
        logits = torch.tanh(h @ W2 + b2)
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs[0], num_samples=1, generator=g).item()
        if idx == 0:
            break
        else:
            context = context[1:] + [idx]
            out.append(itos[idx])

    print(''.join(out))
