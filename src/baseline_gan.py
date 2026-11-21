"""
baseline_gan.py - Text-Conditional GAN baseline (Reed et al., ICML 2016).

Implements a text-conditional DCGAN generating 64x64 face images conditioned
on text prompt embeddings.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class TextEmbeddingNet(nn.Module):
    """
    Lightweight text encoder to embed token sequences into a fixed-size vector.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, seq_len]
        emb = self.embedding(tokens)  # [B, seq_len, embed_dim]
        _, h_n = self.gru(emb)  # h_n: [2, B, hidden_dim]
        out = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, hidden_dim*2]
        return self.relu(self.fc(out))  # [B, embed_dim]


class ConditionalGenerator(nn.Module):
    """
    Conditional DCGAN Generator generating 64x64 RGB images from noise (z) and text embedding.
    """
    def __init__(self, z_dim: int = 100, text_dim: int = 128, ngf: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.text_dim = text_dim
        in_dim = z_dim + text_dim

        self.net = nn.Sequential(
            # Input: [B, z_dim + text_dim, 1, 1] -> [B, ngf*8, 4, 4]
            nn.ConvTranspose2d(in_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # [B, ngf*8, 4, 4] -> [B, ngf*4, 8, 8]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # [B, ngf*4, 8, 8] -> [B, ngf*2, 16, 16]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # [B, ngf*2, 16, 16] -> [B, ngf, 32, 32]
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # [B, ngf, 32, 32] -> [B, 3, 64, 64]
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Concatenate noise and text embedding: [B, z_dim + text_dim, 1, 1]
        inp = torch.cat([z, text_emb], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(inp)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator evaluating 64x64 face image given the conditioned text embedding.
    """
    def __init__(self, text_dim: int = 128, ndf: int = 64):
        super().__init__()
        # Image features extractor down to 4x4
        self.img_conv = nn.Sequential(
            # [B, 3, 64, 64] -> [B, ndf, 32, 32]
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, ndf, 32, 32] -> [B, ndf*2, 16, 16]
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, ndf*2, 16, 16] -> [B, ndf*4, 8, 8]
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, ndf*4, 8, 8] -> [B, ndf*8, 4, 4]
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Spatial projection of text embedding to concatenate at 4x4
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final classification layer
        self.classifier = nn.Sequential(
            # (ndf*8 image channels + ndf*8 text channels) = ndf*16
            nn.Conv2d(ndf * 16, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, img: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # img: [B, 3, 64, 64]
        img_feats = self.img_conv(img)  # [B, ndf*8, 4, 4]

        # Replicate text across spatial dimensions [B, ndf*8, 4, 4]
        t_proj = self.text_proj(text_emb).unsqueeze(2).unsqueeze(3)
        t_replicated = t_proj.expand(-1, -1, 4, 4)

        # Concatenate and classify
        combined = torch.cat([img_feats, t_replicated], dim=1)
        return self.classifier(combined).squeeze()  # [B]
