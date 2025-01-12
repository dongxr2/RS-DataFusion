import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules
from transformers import ViTModel, ViTConfig


class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PSPBlock(
                    in_channels,
                    in_channels // len(sizes),
                    size,
                    use_bathcnorm=use_bathcnorm,
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        use_batchnorm=True,
        out_channels=512,
        dropout=0.2,
        transformer_embed_size=512,
        transformer_num_heads=8,
        transformer_num_layers=6,
        transformer_forward_expansion=4,
    ):
        super().__init__()

        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = modules.Conv2dReLU(
            in_channels=encoder_channels[-1] * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.transformer = TransformerBlock(
            embed_size=transformer_embed_size,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            forward_expansion=transformer_forward_expansion,
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.transformer(x)
        x = self.dropout(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=forward_expansion * embed_size,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1)  # Flatten spatial dimensions
        x = x.permute(2, 0, 1)  # Transformer expects (S, N, E)

        transformer_output = self.transformer(x, x)
        transformer_output = transformer_output.permute(1, 2, 0)  # Back to (N, E, S)
        transformer_output = transformer_output.view(batch_size, channels, height, width)

        return transformer_output