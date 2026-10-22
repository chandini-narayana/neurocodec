# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
ConvLSTM Decoder  — .nec → reconstructed video frames
======================================================
Architecture:
  Input: voxel grid [B × H × W] per timestep
  -> Input projection conv (B -> 32 channels)
  -> 3-layer ConvLSTM stack (32 -> 64 -> 64 -> 32 channels)
  -> Residual refinement block (2x Conv-ReLU-Conv)
  -> Output projection (32 -> 1) + sigmoid
  Output: grayscale frame [1 x H x W], values in [0, 1]

The ConvLSTM state persists between forward() calls so frames can be decoded
one at a time.  Call reset_state() between unrelated sequences.  The state is
NOT detached automatically: during training call reset_state() (or
detach_state()) once per sequence/batch; under torch.no_grad() / inference_mode
nothing accumulates.

Loss: 0.5 x L1 + 0.5 x VGG perceptual loss (torchvision VGG16 features)
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

    def detach_state(self) -> None:
        """Keep the hidden state but cut its autograd history (truncated BPTT)."""
        if self._hidden is not None:
            self._hidden = [(h.detach(), c.detach()) for h, c in self._hidden]

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

        # ConvLSTM (state carries over to the next call; gradients flow through it)
        out, self._hidden = self.convlstm(x, self._hidden)

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
# Single-step wrapper with explicit state (for ONNX / C++ runtime)
# ---------------------------------------------------------------------------

class DecoderStep(nn.Module):
    """
    Stateless view of :class:`ConvLSTMDecoder` for export.

    ``forward(voxel, h0, c0, h1, c1, ...)`` returns ``(frame, h0', c0', h1', c1', ...)``.
    The caller owns the recurrent state, so a runtime without Python can decode
    a clip frame by frame exactly like ``ConvLSTMDecoder.forward`` does.
    Contains no shape-dependent ops, so height and width can be dynamic.
    """

    def __init__(self, decoder: ConvLSTMDecoder) -> None:
        super().__init__()
        self.decoder = decoder

    @property
    def state_channels(self) -> List[int]:
        return [cell.hidden_channels for cell in self.decoder.convlstm.cells]

    def forward(self, voxel: torch.Tensor, *state: torch.Tensor):
        d = self.decoder
        cells = d.convlstm.cells
        if len(state) != 2 * len(cells):
            raise ValueError(f"expected {2 * len(cells)} state tensors, got {len(state)}")
        x = d.input_proj(voxel)
        new_state = []
        for i, cell in enumerate(cells):
            h, c = cell(x, state[2 * i], state[2 * i + 1])
            new_state += [h, c]
            x = h
        frame = d.output_proj(d.refine(x))
        return (frame, *new_state)


def export_onnx(
    model: ConvLSTMDecoder,
    output_path: str | Path,
    height: int = 240,
    width: int = 320,
    opset: int = 17,
) -> List[str]:
    """
    Export ``model`` as a stateful-step ONNX graph (see :class:`DecoderStep`).

    Inputs : ``voxel`` [1,B,H,W], ``h0``, ``c0``, ``h1``, ``c1``, ...
    Outputs: ``frame`` [1,1,H,W], ``h0_out``, ``c0_out``, ...
    Returns the list of input names.
    """
    model = model.eval().cpu()
    step = DecoderStep(model).eval()
    n_bins = model.input_proj[0].in_channels
    voxel = torch.zeros(1, n_bins, height, width)
    state = [torch.zeros(1, ch, height, width) for ch in step.state_channels for _ in (0, 1)]

    in_names, out_names = ["voxel"], ["frame"]
    dyn = {
        "voxel": {0: "batch", 2: "height", 3: "width"},
        "frame": {0: "batch", 2: "height", 3: "width"},
    }
    for i in range(len(step.state_channels)):
        for k in ("h", "c"):
            in_names.append(f"{k}{i}")
            out_names.append(f"{k}{i}_out")
            dyn[f"{k}{i}"] = {0: "batch", 2: "height", 3: "width"}
            dyn[f"{k}{i}_out"] = {0: "batch", 2: "height", 3: "width"}

    kwargs = dict(input_names=in_names, output_names=out_names,
                  dynamic_axes=dyn, opset_version=opset)
    try:  # newer torch defaults to the dynamo exporter; use the TorchScript one
        torch.onnx.export(step, (voxel, *state), str(output_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(step, (voxel, *state), str(output_path), **kwargs)
    return in_names


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
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).to(device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # The frozen feature extractor stays in eval mode.
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # VGG expects ImageNet-normalised 3-channel RGB; repeat grayscale
        return (x.repeat(1, 3, 1, 1) - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.feature_extractor(self._prep(pred)),
                         self.feature_extractor(self._prep(target)))


class CombinedLoss(nn.Module):
    """alpha x L1 + (1 - alpha) x VGG perceptual loss (alpha = 0.5 by default)."""

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
    require_weights: bool = False,
    strict: bool = True,
) -> ConvLSTMDecoder:
    """
    Load a ConvLSTMDecoder with optional pretrained weights.

    If no weights_path is given the bundled ``weights/decoder_v1.pt`` is used
    when it exists.  If no weights file is found, a randomly initialised model
    is returned (with a warning) unless ``require_weights`` is set, in which
    case ``FileNotFoundError`` is raised.  An explicitly given path that does
    not exist is always an error.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
    if weights_path and not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    ckpt = None
    if path.exists():
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
    elif require_weights:
        raise FileNotFoundError(
            f"No decoder weights at {path}. Train them with training/train_decoder.ipynb "
            "or pass --weights; see neurocodec/weights/README.md.")

    hidden_dims = None
    if isinstance(ckpt, dict):
        hidden_dims = (ckpt.get("config") or {}).get("hidden_dims")
    model = ConvLSTMDecoder(n_bins=n_bins, hidden_dims=hidden_dims)

    if ckpt is not None:
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=strict)
        logger.info("Loaded weights from %s", path)
    else:
        logger.warning(
            "No weights found at %s - using random initialisation (output is meaningless). "
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
    allow_random_weights: bool = False,
) -> dict:
    """
    Decode a .nec file to a reconstructed grayscale video.

    Parameters
    ----------
    nec_path    : str | Path  Input .nec file.
    output_path : str | Path  Output video path (e.g. "out.mp4").
    weights_path: str | Path  Optional path to model weights (.pt).
    device      : str         'cpu', 'cuda', or None (auto).
    allow_random_weights : bool
        Without trained weights the output is a meaningless constant image, so
        by default a missing weights file raises ``FileNotFoundError``.  Set
        this to True to run the pipeline with a random model (for testing).

    Returns
    -------
    dict  Keys: n_frames, fps, output_path, device
    """
    nec_path    = Path(nec_path)
    output_path = Path(output_path)

    _device = torch.device(device) if device else \
              torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = NECReader(nec_path)
    model  = load_decoder(weights_path, n_bins=reader.n_bins, device=_device,
                          require_weights=not allow_random_weights)
    model.reset_state()

    H, W = reader.height, reader.width
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, reader.fps, (W, H), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path}")

    n = 0
    try:
        with torch.inference_mode():
            for voxel in reader.iter_dense():
                x     = torch.from_numpy(voxel).unsqueeze(0).to(_device)   # (1, B, H, W)
                pred  = model(x)                                             # (1, 1, H, W)
                frame = (pred[0, 0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                writer.write(frame)
                n += 1
    finally:
        writer.release()
    logger.info("Decoded %d frames -> %s", n, output_path)
    return {
        "n_frames":    n,
        "fps":         reader.fps,
        "output_path": str(output_path),
        "device":      str(_device),
    }
