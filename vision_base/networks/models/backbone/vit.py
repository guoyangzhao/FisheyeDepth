"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np

def load_pretrained_weights(
    model, 
    model_name=None, 
    weights_path=None, 
    load_first_conv=True, 
    load_fc=True, 
    load_repr_layer=False,
    resize_positional_embedding=False,
    target_shape=None,
    verbose=True,
    strict=False,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    
    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias', 'norm.weight', 'norm.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding: 
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_pos_embed(pos_embed=posemb, target_shape=target_shape, pos_shape=(224 // model.patches, 224 // model.patches), patch_size=model.patches)
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return (x, x) if isinstance(x, int) else x


def resize_pos_embed(pos_embed, target_shape, pos_shape, patch_size):
    """Resize pos_embed weights.
    Resize pos_embed using bicubic interpolate method.
    Args:
        pos_embed (torch.Tensor): pos_embed weights.
        input_shape (tuple): Tuple for (input_h, intput_w).
        pos_shape (tuple): Tuple for (pos_h, pos_w).
        patch_size (int): Patch size.
    Return:
        torch.Tensor: The resized pos_embed of shape [B, L_new, C]
    """
    print(patch_size)
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
    input_h, input_w = target_shape
    pos_h, pos_w = pos_shape
    cls_token_weight = pos_embed[:, 0]
    pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w ):]
    pos_embed_weight = pos_embed_weight.reshape(
        1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
    pos_embed_weight = F.interpolate(
        pos_embed_weight,
        size=[input_h // patch_size, input_w // patch_size],
        align_corners=False,
        mode='bicubic')
    cls_token_weight = cls_token_weight.unsqueeze(1)
    pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
    pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
    return pos_embed

def get_base_config():
    """Base ViT config ViT"""
    return dict(
      dim=768,
      ff_dim=3072,
      num_heads=12,
      num_layers=12,
      attention_dropout_rate=0.0,
      dropout_rate=0.1,
      representation_size=768,
      classifier='token'
    )

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(patches=(16, 16)))
    return config

def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patches=(32, 32)))
    return config

def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        ff_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        representation_size=1024
    ))
    return config

def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patches=(32, 32)))
    return config

def drop_head_variant(config):
    config.update(dict(representation_size=None))
    return config


PRETRAINED_MODELS = {
    'B_16': {
      'config': get_b16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth"
    },
    'B_32': {
      'config': get_b32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32.pth"
    },
    'L_16': {
      'config': get_l16_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': None
    },
    'L_32': {
      'config': get_l32_config(),
      'num_classes': 21843,
      'image_size': (224, 224),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32.pth"
    },
    'B_16_imagenet1k': {
      'config': drop_head_variant(get_b16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth"
    },
    'B_32_imagenet1k': {
      'config': drop_head_variant(get_b32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_32_imagenet1k.pth"
    },
    'L_16_imagenet1k': {
      'config': drop_head_variant(get_l16_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_16_imagenet1k.pth"
    },
    'L_32_imagenet1k': {
      'config': drop_head_variant(get_l32_config()),
      'num_classes': 1000,
      'image_size': (384, 384),
      'url': "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/L_32_imagenet1k.pth"
    },
}




def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask, positional_embedding_attn=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if positional_embedding_attn is not None:
            scores = positional_embedding_attn(scores)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores.detach()
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, positional_embedding_attn=None):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask, positional_embedding_attn)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    #def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
    def __init__(self, seq_len, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        self.positional_embedding_attn = PositionalEmbedding1D(seq_len, seq_len)
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    #def forward(self, x, positional_embedding, mask=None):
    def forward(self, x, mask=None):
        for block in self.blocks:
            #x = positional_embedding(x)
            x = self.positional_embedding(x)
            x = block(x, mask, self.positional_embedding_attn)
        return x


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding

class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        pretrained_name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.patches = patches
        # Configuration
        if pretrained_name is None:
            pass
        else:  # load pretrained model
            assert pretrained_name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            # config = PRETRAINED_MODELS[pretrained_name]['config']

        self.image_size = image_size                

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
        
        # Transformer
        #self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
        self.transformer = Transformer(seq_len, num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        
        # Initialize weights
        self.init_weights()
        
        # Load pretrained model
        if pretrained:
            assert isinstance(pretrained_name, str)
            pretrained_num_channels = 3
            # pretrained_num_classes = PRETRAINED_MODELS[pretrained_name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[pretrained_name]['image_size']
            load_pretrained_weights(
                self, pretrained_name, 
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=False,
                load_repr_layer=False,
                target_shape=self.image_size,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 
        #x = self.transformer(x, self.positional_embedding)  # b,gh*gw+1,d
        x = self.transformer(x)  # b,gh*gw+1,d
        x = x[:, 1:]
        out = x.reshape(b, fh // self.patches, fw // self.patches, -1).permute(0, 3, 1, 2).contiguous()

        return [out]
