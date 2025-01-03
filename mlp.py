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
C = torch.randn((27, 50), generator=g, requires_grad=True)
W1 = torch.randn((150, 300), generator=g, requires_grad=True)
b1 = torch.randn((300,), generator=g, requires_grad=True)
W2 = torch.randn((300, 27), generator=g, requires_grad=True)
b2 = torch.randn((27,), generator=g, requires_grad=True)

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



"""
Example output:

lrao
dra
zelzynahyt
usieo
lshy
leo
inshgdellyajtin
isliganeplaj
eeniezidns
llebfah
"""
