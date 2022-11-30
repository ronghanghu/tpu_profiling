from functools import partial

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import timm.models.vision_transformer


class AttentionWithEinsum(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_param = nn.Parameter(torch.zeros(dim, num_heads, head_dim))
        self.k_param = nn.Parameter(torch.zeros(dim, num_heads, head_dim))
        self.v_param = nn.Parameter(torch.zeros(dim, num_heads, head_dim))
        self.p_param = nn.Parameter(torch.zeros(num_heads, head_dim, dim))
        nn.init.uniform_(self.q_param, -(dim ** -0.5), dim ** -0.5)
        nn.init.uniform_(self.k_param, -(dim ** -0.5), dim ** -0.5)
        nn.init.uniform_(self.v_param, -(dim ** -0.5), dim ** -0.5)
        nn.init.uniform_(self.p_param, -(dim ** -0.5), dim ** -0.5)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q = torch.einsum("bsc,chd->bshd", x, self.q_param) * self.scale
        k = torch.einsum("btc,chd->bthd", x, self.k_param)
        v = torch.einsum("btc,chd->bthd", x, self.v_param)
        attn = torch.einsum("bshd,bthd->bhst", q, k).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum("bhst,bthd->bshd", attn, v)
        x = torch.einsum("bshd,hdc->bsc", x, self.p_param)
        x = self.proj_drop(x)
        return x


class VisionTransformer(nn.Module):
    """ VisionTransformer. """

    def __init__(
        self,
        config,
        num_classes,
        patch_size,
        hidden_size,
        num_layers,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if config.use_attention_with_einsum:
            xm.master_print("using ViT self-attention with einsum")
            timm.models.vision_transformer.Attention = AttentionWithEinsum

        act_layer = nn.GELU
        if config.use_tanh_for_gelu:
            xm.master_print("approximate='tanh' in nn.GELU")
            act_layer = partial(nn.GELU, approximate="tanh")

        # --------------------------------------------------------------------------
        # ViT encoder specifics
        image_size = config.image_size
        self.patch_embed = timm.models.vision_transformer.PatchEmbed(image_size, patch_size, 3, hidden_size)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))

        self.blocks = nn.ModuleList(
            [
                timm.models.vision_transformer.Block(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm_layer(hidden_size)
        self.num_classes = num_classes
        if self.num_classes:
            self.head = nn.Linear(hidden_size, self.num_classes)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.pos_embed, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed

        # apply Transformer blocks
        for n, blk in enumerate(self.blocks):
            with xp.Trace(f"encoder_{n}"):
                x = blk(x)
        x = self.norm(x)

        if self.num_classes:
            x = self.head(x[:, 0])
        return x


ViT_B16_nodrop = partial(
    VisionTransformer,
    patch_size=16,
    hidden_size=768,
    num_layers=12,
    mlp_ratio=4.0,
    num_heads=12,
    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
)
ViT_L16_nodrop = partial(
    VisionTransformer,
    patch_size=16,
    hidden_size=1024,
    num_layers=24,
    mlp_ratio=4.0,
    num_heads=16,
    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
)
ViT_H14_nodrop = partial(
    VisionTransformer,
    patch_size=14,
    hidden_size=1280,
    num_layers=32,
    mlp_ratio=4.0,
    num_heads=16,
    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
)
