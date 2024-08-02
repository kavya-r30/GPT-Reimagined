import torch
import torch.nn as nn
from torch.nn import functional as F

#parameter list, main knobs lie here
block_size = 8 #this is T
batch_size = 32 #this is B
max_iters = 3000
interval = 100
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64 #C
########################

with open('marvel.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda lst: ''.join([itos[i] for i in lst])
data = torch.tensor(encode(text),dtype=torch.long,device=device)

train_len = int(0.9*len(data))
train_data = data[:train_len]
val_data = data[train_len:]

block_size = 8
batch_size = 4

def get_batch(splittype):
    data = val_data
    if(splittype=='train'): 
        data = train_data
    ix = torch.randint(len(data)-block_size,(batch_size,),device=device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    print(out)
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd,device=device) 
        self.positional_embedding_table = nn.Embedding(block_size,n_embd,device=device)
        self.lm_head = nn.Linear(n_embd, vocab_size,device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B x T x C
        pos_emb = self.positional_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)  # B x T x vocab_size
        if targets is None:
            loss = None
        else:
            C = logits.shape[-1]
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx[:,-block_size:])
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel()
model = m.to(device=device)

optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

for iter in range(max_iters):
    if iter%interval==0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb,yb = get_batch('train')
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(context,max_new_tokens=block_size+500)[0].tolist()))