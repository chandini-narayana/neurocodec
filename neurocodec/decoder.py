# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
ConvLSTM Decoder  — .nec → reconstructed video frames
======================================================
Architecture:
  Input: voxel grid [B × H × W] per timestep
  ↓ Input projection conv (B → 32 channels)
  ↓ 3-layer ConvLSTM stack (32 → 64 → 64 → 32 channels)
  ↓ Residual refinement block (2× Conv-ReLU-Conv)
  ↓ Output projection (32 → 1) + sigmoid
  Output: grayscale frame [1 × H × W], values in [0, 1]

Loss: 0.5 × L1 + 0.5 × VGG perceptual loss (torchvision VGG16 features)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurocodec.convlstm import ConvLSTMStack
from neurocodec.encoder import NECReader

logger = logging.getLogger(__name__)

# Default model weights path (relative to package)
_DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "decoder_v1.pt"


# ---------------------------------------------------------------------------
# Residual refinement block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ---------------------------------------------------------------------------
# Decoder network
# ---------------------------------------------------------------------------

class ConvLSTMDecoder(nn.Module):
    """
    ConvLSTM-based video frame reconstructor.

    Parameters
    ----------
    n_bins      : int   Number of voxel time bins (= input channels).
    hidden_dims : list  Hidden channel dimensions for each ConvLSTM layer.
    """

    def __init__(
        self,
        n_bins: int = 5,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        # Input projection: map B voxel bins to a feature volume
        self.input_proj = nn.Sequential(
            nn.Conv2d(n_bins, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM stack
        self.convlstm = ConvLSTMStack(
            in_channels=32,
            hidden_channels=hidden_dims,
            kernel_size=3,
        )

        feat_ch = hidden_dims[-1]

        # Residual refinement
        self.refine = nn.Sequential(
            ResBlock(feat_ch),
            ResBlock(feat_ch),
        )

        # Output projection → 1-channel grayscale
        self.output_proj = nn.Sequential(
            nn.Conv2d(feat_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self._hidden: Optional[List[Tuple]] = None
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Reset ConvLSTM hidden state (call between unrelated sequences)."""
        self._hidden = None

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        voxel : (B_batch, n_bins, H, W)  or  (B_batch, T_seq, n_bins, H, W)
                Single frame or sequence of voxel grids.

        Returns
        -------
        torch.Tensor  (B_batch, 1, H, W)  or  (B_batch, T_seq, 1, H, W)
        """
        single_frame = voxel.ndim == 4
        if single_frame:
            voxel = voxel.unsqueeze(1)  # add time dim

        B_batch, T, C, H, W = voxel.shape

        # Project each timestep's voxel
        x = voxel.view(B_batch * T, C, H, W)
        x = self.input_proj(x)
        x = x.view(B_batch, T, -1, H, W)  # (B, T, 32, H, W)

        # ConvLSTM
        out, self._hidden = self.convlstm(x, self._hidden)
        # Detach hidden to prevent BPTT across independent sequences
        self._hidden = [(h.detach(), c.detach()) for h, c in self._hidden]

        # Decode each timestep
        out = out.view(B_batch * T, -1, H, W)
        out = self.refine(out)
        out = self.output_proj(out)
        out = out.view(B_batch, T, 1, H, W)

        if single_frame:
            out = out.squeeze(1)
        return out

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# VGG Perceptual Loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features (relu2_2 layer).
    Encourages texture fidelity beyond pixel-level L1.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).to(device)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # VGG expects 3-channel RGB; repeat grayscale
        pred_rgb   = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        feat_pred   = self.feature_extractor(pred_rgb)
        feat_target = self.feature_extractor(target_rgb)
        return F.l1_loss(feat_pred, feat_target)


class CombinedLoss(nn.Module):
    """0.5 × L1 + 0.5 × VGG perceptual loss."""

    def __init__(self, device: torch.device, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha   = alpha
        self.l1      = nn.L1Loss()
        self.percept = VGGPerceptualLoss(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.l1(pred, target) + (1 - self.alpha) * self.percept(pred, target)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_decoder(
    weights_path: Optional[str | Path] = None,
    n_bins: int = 5,
    device: Optional[torch.device] = None,
) -> ConvLSTMDecoder:
    """
    Load a ConvLSTMDecoder with optional pretrained weights.

    If no weights_path is given and the default bundled weights exist,
    those are loaded automatically.  Otherwise returns a freshly
    initialised model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvLSTMDecoder(n_bins=n_bins)

    path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
    if path.exists():
        ckpt = torch.load(str(path), map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded weights from %s", path)
    else:
        logger.warning(
            "No weights found at %s — using random initialisation. "
            "Run training/train_decoder.ipynb on Colab to train the model.",
            path,
        )

    model.to(device).eval()
    return model


def decode(
    nec_path: str | Path,
    output_path: str | Path,
    weights_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Decode a .nec file to a reconstructed video.

    Parameters
    ----------
    nec_path    : str | Path  Input .nec file.
    output_path : str | Path  Output video path (e.g. "out.mp4").
    weights_path: str | Path  Optional path to model weights (.pt).
    device      : str         'cpu', 'cuda', or None (auto).

    Returns
    -------
    dict  Keys: n_frames, fps, output_path, device
    """
    nec_path    = Path(nec_path)
    output_path = Path(output_path)

    _device = torch.device(device) if device else \
              torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = NECReader(nec_path)
    model  = load_decoder(weights_path, n_bins=reader.n_bins, device=_device)
    model.reset_state()

    dense = reader.to_dense()  # (T, B, H, W) float32
    T, B, H, W = dense.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (W, H), isColor=False)

    with torch.no_grad():
        for t in range(T):
            voxel = torch.from_numpy(dense[t]).unsqueeze(0).to(_device)  # (1, B, H, W)
            pred  = model(voxel)                                           # (1, 1, H, W)
            frame = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            writer.write(frame)

    writer.release()
    logger.info("Decoded %d frames → %s", T, output_path)
    return {
        "n_frames":   T,
        "fps":        30.0,
        "output_path": str(output_path),
        "device":      str(_device),
    }
