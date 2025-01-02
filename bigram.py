import torch
import torch.nn.functional as F

# The simplest character level language model (a bigram model)

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0

# Create the training set
xs = []
ys = []
num_examples = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]; idx2 = stoi[ch2]
        xs.append(idx1); ys.append(idx2)
        num_examples += 1

xs = torch.tensor(xs); ys = torch.tensor(ys)

xs_enc = F.one_hot(xs, num_classes=27).float()
ys_enc = F.one_hot(ys, num_classes=27)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
print(xs.shape)

for epoch in range(100):
    logits = xs_enc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = (-probs[torch.arange(num_examples), ys].log()).mean()
    print(loss.item())
    W.grad = None
    loss.backward()
    W.data = W.data - 1 * W.grad


# Sampling a few words

idx = None
out = None
for _ in range(10):
    idx = 0
    out = []
    while True:
        x_enc = F.one_hot(torch.tensor([idx]), 27).float()
        logits = x_enc @ W
        counts = logits.exp()
        p = counts / counts.sum(dim=1, keepdim=True)

        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if idx == 0:
            break
        else:
            out.append(chr(idx - 1 + 97))
    print(''.join(out))


"""
Example output:
edestlelengiyasholi
der
bitan
jela
r
xynte
je
bqfrioghuiela
tarimpananaum
lileyn
"""
