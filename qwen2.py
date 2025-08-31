from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from transformers import Qwen2Model
from transformers import AutoTokenizer, AutoModelForCausalLM

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

@dataclass
class QwenConfig:
    vocab_size: int = 151936
    n_layer: int = 24
    q_heads: int = 14
    kv_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    n_embd: int = 896
    block_size: int = 131072

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class Qwen2RMSNorm(nn.Module):
    """
    Custom RMSNorm implementation matching HuggingFace's Qwen2RMSNorm
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.head_dim = config.head_dim
        self.q_heads = config.q_heads
        self.kv_heads = config.kv_heads
        self.group_size = self.q_heads // self.kv_heads
        self.block_size = config.block_size

        self.q_proj = nn.Linear(config.n_embd, config.q_heads * config.head_dim, bias=True)
        self.k_proj = nn.Linear(config.n_embd, config.kv_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.n_embd, config.kv_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.q_heads * config.head_dim, config.n_embd, bias=False)

        base = 1000000.0  
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))        
        position = torch.arange(self.block_size, dtype=torch.float)        
        sinusoid_inp = torch.outer(position, inv_freq)
        cos_half = sinusoid_inp.cos()  
        sin_half = sinusoid_inp.sin()  
        cos = torch.stack([cos_half, cos_half], dim=-1).flatten(start_dim=-2)  # [seq_len, head_dim]
        sin = torch.stack([sin_half, sin_half], dim=-1).flatten(start_dim=-2)  # [seq_len, head_dim]
        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        cos = self.cos[:T]
        sin = self.sin[:T]

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)        
        self.input_layernorm = Qwen2RMSNorm(config.n_embd, eps=1e-6)
        self.post_attention_layernorm = Qwen2RMSNorm(config.n_embd, eps=1e-6)
        
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = Qwen2RMSNorm(config.n_embd, eps=1e-06)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.embed_tokens(idx)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import Qwen2Model
        print("loading weights from pretrained qwen: %s" % model_type)
        config_args = {
            'Qwen/Qwen2-0.5B' : dict(n_layer=24, q_heads=14, kv_heads=2)
        }['Qwen/Qwen2-0.5B']

        config = QwenConfig(**config_args)
        model = Qwen(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        model_hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        embed_weight = model_hf.model.embed_tokens.weight
        lm_head_weight = model_hf.lm_head.weight
        embeddings_are_tied = torch.equal(embed_weight, lm_head_weight)
        for k in sd_hf.keys():
            if k == 'lm_head.weight':
                if embeddings_are_tied:
                    print("Skipping lm_head.weight (tied with embeddings)")
                    continue
                else:
                    sd[k].copy_(sd_hf[k])
            else:
                custom_key = k[6:]  
                if custom_key in sd:
                    sd[custom_key].copy_(sd_hf[k])
                    if custom_key == 'embed_tokens.weight' and embeddings_are_tied:
                        print("Loaded embed_tokens.weight (also used for lm_head due to tying)")
                else:
                    print(f"Warning: Key {custom_key} not found in custom model")
        return model
    
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T +1) > len(self.tokens):
            self.current_position = 0
        return x, y


    
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("using device", device)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True)
# model = Qwen.from_pretrained('Qwen/Qwen2-0.5B')
model = Qwen(QwenConfig())
print('success')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=1024)

torch.set_float32_matmul_precision('high')

model.eval()
model.to(device)
model = torch.compile(model)
# logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step: {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
import sys; sys.exit(0)

max_length = 30
num_return_sequences = 5

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True)
tokens = tokenizer.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')

while x.size(1) < 30:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = tokenizer.decode(tokens)
    print(">", decoded)
