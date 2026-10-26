
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


def timestep_embedding(timesteps, dim=64, max_period=10000):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )

    return embedding


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CondNAFBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        expanded = channels * 2

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded
        )

        self.gate = SimpleGate()

        self.attention_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded // 4, expanded // 4, 1),
        )
        self.attention_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(expanded // 4, expanded // 4, 1),
        )

        self.conv3 = nn.Conv2d(expanded // 2, channels, 1)
        self.dropout1 = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.dropout2 = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)

        x_avg, x_max = x.chunk(2, dim=1)
        x_avg = self.attention_avg(x_avg) * x_avg
        x_max = self.attention_max(x_max) * x_max

        x = torch.cat([x_avg, x_max], dim=1)
        x = self.dropout1(self.conv3(x))
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.gate(x)
        x = self.dropout2(self.conv5(x))

        return y + x * self.gamma


class NAFBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        expanded = channels * 2

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded
        )

        self.gate = SimpleGate()

        self.attention_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded // 4, expanded // 4, 1),
        )
        self.attention_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(expanded // 4, expanded // 4, 1),
        )

        self.conv3 = nn.Conv2d(expanded // 2, channels, 1)
        self.dropout1 = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, channels),
        )

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.dropout2 = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp, time_embedding):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)

        x_avg, x_max = x.chunk(2, dim=1)
        x_avg = self.attention_avg(x_avg) * x_avg
        x_max = self.attention_max(x_max) * x_max

        x = torch.cat([x_avg, x_max], dim=1)
        x = self.dropout1(self.conv3(x))
        y = inp + x * self.beta

        y = y + self.time_emb(time_embedding)[:, :, None, None]

        x = self.conv4(self.norm2(y))
        x = self.gate(x)
        x = self.dropout2(self.conv5(x))

        return y + x * self.gamma


class EmbedSequential(nn.Sequential):
    def forward(self, x, time_embedding):
        for layer in self:
            if isinstance(layer, NAFBlock):
                x = layer(x, time_embedding)
            else:
                x = layer(x)
        return x


class DiffCRModel(nn.Module):
    """
    Single-condition DiffCR adaptation for 3-band LISS-IV patches.

    Input channels:
        0:3 = cloudy image
        3:6 = noisy clear-target image

    Output:
        3 channels representing the predicted clean image.
    """

    def __init__(
        self,
        img_channels=3,
        width=32,
        enc_blk_nums=(1, 1, 1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1, 1, 1),
        dropout=0.0,
    ):
        super().__init__()

        self.img_channels = img_channels

        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)
        self.cond_intro = nn.Conv2d(img_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        self.encoders = nn.ModuleList()
        self.cond_encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.cond_downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width

        for num_blocks in enc_blk_nums:
            self.encoders.append(
                EmbedSequential(
                    *[
                        NAFBlock(channels, dropout)
                        for _ in range(num_blocks)
                    ]
                )
            )
            self.cond_encoders.append(
                nn.Sequential(
                    *[
                        CondNAFBlock(channels, dropout)
                        for _ in range(num_blocks)
                    ]
                )
            )

            self.downs.append(
                nn.Conv2d(channels, channels * 2, 2, stride=2)
            )
            self.cond_downs.append(
                nn.Conv2d(channels, channels * 2, 2, stride=2)
            )

            channels *= 2

        self.middle = EmbedSequential(
            *[
                NAFBlock(channels, dropout)
                for _ in range(middle_blk_num)
            ]
        )

        for num_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )

            channels //= 2

            self.decoders.append(
                EmbedSequential(
                    *[
                        NAFBlock(channels, dropout)
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp, timesteps):
        if inp.ndim != 4:
            raise ValueError("Input must have shape (B, C, H, W).")

        expected_channels = self.img_channels * 2
        if inp.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} input channels "
                f"(cloudy + noisy target), got {inp.shape[1]}."
            )

        cloudy, noisy_target = inp.split(self.img_channels, dim=1)

        h, w = inp.shape[-2:]
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size

        noisy_target = F.pad(noisy_target, (0, pad_w, 0, pad_h))
        cloudy = F.pad(cloudy, (0, pad_w, 0, pad_h))

        time_embedding = self.time_mlp(
            timestep_embedding(timesteps, dim=64)
        )

        x = self.intro(noisy_target)
        cond = self.cond_intro(cloudy)

        skips = []

        for encoder, cond_encoder, down, cond_down in zip(
            self.encoders,
            self.cond_encoders,
            self.downs,
            self.cond_downs,
        ):
            x = encoder(x, time_embedding)
            cond = cond_encoder(cond)

            x = x + cond
            skips.append(x)

            x = down(x)
            cond = cond_down(cond)

        x = self.middle(x, time_embedding)

        for decoder, up, skip in zip(
            self.decoders, self.ups, reversed(skips)
        ):
            x = up(x)
            x = x + skip
            x = decoder(x, time_embedding)

        x = self.ending(x)

        return x[..., :h, :w]
