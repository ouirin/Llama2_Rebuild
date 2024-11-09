import math
import sys

import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass


@dataclass
class Llama2Config():

    hidden_size: int = 4096
    n_head: int = 32
    n_kv_head: int = 32
    num_layers: int = 32

    multiple_of: int = 256
    ffn_dim_multiplier: float = None

    vocab_size: int = 32000
    layernorm_epsilon: float = 1e-5
    max_seq_len: int = 4096


def precompute_rotary_complex(dim, length, theta: float = 10000.0):

    denominator = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    angle = torch.outer(torch.arange(length).float(), denominator)
    rotary_complex = torch.polar(torch.ones_like(angle), angle)

    return rotary_complex


def reshape_for_broadcast(rotary_complex, x):

    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return rotary_complex.view(*shape)


def apply_rotary_emb(q, k, rotary_complex):

    q_ = torch.view_as_complex(q.reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.reshape(*k.shape[:-1], -1, 2))

    rotary_complex = reshape_for_broadcast(rotary_complex, q_)

    q_out = torch.view_as_real(q_ * rotary_complex).flatten(-2)
    k_out = torch.view_as_real(k_ * rotary_complex).flatten(-2)

    return q_out.type_as(q), k_out.type_as(k)


class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, dtype=None):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype))
        self.eps = eps

    def _norm(self, x):

        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):

        output = self._norm(x.float()).type_as(x)

        return output * self.weight


class LlamaRMSNorm(nn.Module):

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


class Llama2Attention(nn.Module):

    def __init__(self, n_state, n_head, n_kv_head, layer_idx, dtype=None):

        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_state // n_head
        self.q_proj = Linear(n_state, n_head * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = Linear(n_state, n_kv_head * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = Linear(n_state, n_kv_head * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = Linear(n_head * self.head_dim, n_state, bias=False, dtype=dtype)

    def forward(self, x, rotary_complex, attention_mask, kv_cache):

        n_batch, n_seq, _ = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.view(n_batch, n_seq, self.n_head, self.head_dim)
        k = k.view(n_batch, n_seq, self.n_kv_head, self.head_dim)
        v = v.view(n_batch, n_seq, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, rotary_complex=rotary_complex)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        kv_cache = (k.detach(), v.detach())

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        qk = torch.matmul(q, k) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            qk = qk + attention_mask
        scores = F.softmax(qk.float(), dim=-1).type_as(x)

        output = torch.matmul(scores, v)
        output = output.permute(0, 2, 1, 3).reshape(n_batch, n_seq, -1)
        output = self.o_proj(output)

        return output, kv_cache


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, multiple_of, dtype):

        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.up_proj = Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.down_proj = Linear(hidden_dim, dim, bias=False, dtype=dtype)

    def forward(self, x):

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Llama2Block(nn.Module):

    def __init__(self, layer_idx, config, dtype=None):

        super().__init__()
        self.layer_idx = layer_idx
        self.attn_ln = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.attn = Llama2Attention(config.hidden_size, config.n_head, config.n_kv_head, layer_idx, dtype=dtype)
        self.ffn_ln = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.ffn = FeedForward(config.hidden_size, hidden_dim=4*config.hidden_size, multiple_of=config.multiple_of, dtype=dtype)

    def forward(self, x, rotary_complex, attention_mask, kv_cache):

        h, kv_cache = self.attn(self.attn_ln(x), rotary_complex, attention_mask, kv_cache)
        x = x + h
        h = self.ffn(self.ffn_ln(x))
        output = x + h

        return output, kv_cache


class Llama2Model(nn.Module):

    def __init__(self, config, dtype):

        super().__init__()
        self.config = config
        self.word_embedding = Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList([Llama2Block(layer_idx, config, dtype=dtype) for layer_idx in range(config.num_layers)])
        self.final_ln = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        rotary_complex = precompute_rotary_complex(config.hidden_size//config.n_head, config.max_seq_len)
        self.register_buffer("rotary_complex", rotary_complex, persistent=False)

    def prepare_input(self, input_ids, all_kv_cache):

        device = input_ids.device

        n_batch, n_seq_new = input_ids.shape

        if all_kv_cache is not None:
            n_seq_past = all_kv_cache[0][0].shape[1]
            n_seq = n_seq_new + n_seq_past
        else:
            n_seq = n_seq_new

        input_embeddings = self.word_embedding(input_ids)

        attention_mask = None
        if n_seq_new > 1:
            attention_mask = torch.full((n_seq, n_seq), float("-inf"), device=device)
            attention_mask = torch.triu(attention_mask, diagonal=1).type_as(input_embeddings)

        if all_kv_cache is not None:
            rotary_complex = self.rotary_complex[n_seq-1]
        else:
            rotary_complex = self.rotary_complex[0:n_seq]

        return input_embeddings, attention_mask, rotary_complex

    def forward(self, input_ids, all_kv_cache):

        input_embeddings, attention_mask, rotary_complex = self.prepare_input(input_ids, all_kv_cache)

        h = input_embeddings

        current_kv = tuple()

        for i, layer in enumerate(self.layers):

            kv_cache = all_kv_cache[i] if all_kv_cache is not None else None

            h, kv_cache = layer(h, rotary_complex=rotary_complex, attention_mask=attention_mask, kv_cache=kv_cache)

            current_kv += (kv_cache,)

        h = self.final_ln(h)

        output = self.lm_head(h)

        loss = None

        return loss, output, current_kv


class Linear(nn.Linear):

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), None if self.bias is None else self.bias.type_as(x))

    def reset_parameters(self):
        pass


class Embedding(nn.Embedding):

    def reset_parameters(self):
        pass

    # try to batch infer logit, but don't work
    # def forward(self, input_ids):
    #
    #     mask = input_ids == -1
    #
    #     input_ids[mask] = 0
    #
    #     embeddings = super().forward(input_ids)
    #
    #     # embeddings[mask] = 0
    #
    #     return embeddings
