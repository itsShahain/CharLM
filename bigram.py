import torch

# The simplest character level language model (a bigram model)

words = open("names.txt", "r").read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]; idx2 = stoi[ch2]
        N[idx1, idx2] += 1
 g = torch.Generator().manual_seed(2147483647)
P = N / N.sum(dim=1, keepdim=True)

for _ in range(10):
    idx = 0
    out = []
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if idx == 0:
            break
        else:
            out.append(chr(idx - 1 + 97))
    print(''.join(out))


"""
Example output:
junide
janasah
p
cony
a
nn
kohin
tolian
juee
ksahnaauranilevias
"""
