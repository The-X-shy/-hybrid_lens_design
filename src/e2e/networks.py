"""
Network components for end-to-end hybrid lens training.
端到端混合透镜训练网络组件
"""

from typing import Optional

import torch
import torch.nn as nn

from deeplens.network.reconstruction.unet import UNet


class SpatialMHABlock(nn.Module):
    """
    Apply MHA on flattened spatial tokens then project back to image features.
    对空间 token 应用多头注意力，再投影回图像特征
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 2

        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        feat = self.proj_in(x)  # [B, E, H, W]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, HW, E]

        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn_out)

        ffn_out = self.ffn(tokens)
        tokens = self.norm2(tokens + ffn_out)

        feat = tokens.transpose(1, 2).reshape(b, -1, h, w)
        return x + self.proj_out(feat)


class MHAUNet(nn.Module):
    """
    MHA front-end + UNet reconstruction back-end.
    前置多头注意力模块 + UNet 重建模块
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mha = SpatialMHABlock(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mha(x)
        return self.unet(x)
