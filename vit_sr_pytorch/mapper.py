from operator import index
import torch
from torch import Tensor, nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat
from backbone import NestedTensor, Backbone

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def apply_pos_emb(pos_emb, qkv):
    n = qkv[0].shape[-2]
    pos_emb = pos_emb[..., :n, :]
    return tuple(map(lambda t: apply_rotary_emb(pos_emb, t), qkv))


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, index_dim: tuple = (256, 256), num_channels: int = 256):
        super().__init__()
        self.height = index_dim[0]
        self.width = index_dim[1]
        self.row_embed = nn.Parameter(num_channels // 2, self.height)
        self.col_embed = nn.Parameter(num_channels // 2, self.width)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, batch_size):
        pos = torch.cat([
            self.col_embed.unsqueeze(0).repeat(self.height, 1, 1),
            self.row_embed.repeat(1, self.width, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return pos


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(
            context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, rotary_pos_emb=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if exists(rotary_pos_emb):
            k, v = apply_pos_emb(rotary_pos_emb, (k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # TODO: Check mask is applied correctly
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Mapper(nn.Module):
    """Maps feature maps onto a latent space."""

    def __init__(self,
                 backbone,
                 fmap_size,
                 output_size,
                 latent_dim: int = 256,
                 features_dim: int = 960,
                 dim_head: int = 64,
                 heads=8,
                 dropout=0.,
                 attn_mlp: bool = False):
        self.latents = PositionEmbeddingLearned(
            index_dim=output_size, num_channels=latent_dim)
        self.backbone = backbone

        axial_pos_emb = RotaryEmbedding(dim=dim_head // 3, freqs_for='pixel')
        img_freqs_axial = axial_pos_emb(torch.linspace(-1, 1, steps=fmap_size))
        img_freqs = broadcat((rearrange(img_freqs_axial, 'i d -> i () d'),
                             rearrange(img_freqs_axial, 'j d -> () j d')), dim=-1)
        img_freqs = rearrange(img_freqs, 'h w d -> (h w) d')
        self.register_buffer('pos_emb', img_freqs)

        self.attention = PreNorm(latent_dim, Attention(
            latent_dim, context_dim=features_dim, dim_head=dim_head, heads=heads, dropout=dropout))
        self.mlp = PreNorm(latent_dim, FeedForward(
            latent_dim)) if attn_mlp else None

    def forward(self, x: Tensor):
        features: NestedTensor = self.backbone(x)
        x = self.attention(self.latents(x.shape[0]), features, self.pos_emb)

        if exists(self.mlp):
            x = self.mlp(x) + x
        return x
