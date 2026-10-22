# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
Voxel-Grid Encoder  ->  .nec binary format
==========================================
Pipeline:
  video -> DVS events -> per-frame voxel grid [B x H x W] (sliding window of
  the last B frames) -> top-K sparsification -> int8 quantisation ->
  delta-coded indices -> DEFLATE (LZ77 + Huffman) -> .nec file

Voxel grid
----------
The voxel for frame ``t`` has ``B`` temporal bins; bin ``k`` accumulates the
signed events of frame ``t - (B - 1) + k`` (so the last bin is the current
frame and earlier bins give the decoder short-term motion context).  Frames
before the start of the clip contribute empty bins.  Each voxel is
L-infinity-normalised to [-1, 1].

.nec file format, version 2 (little-endian)
-------------------------------------------
Offset  Size  Type    Field
------  ----  ------  -----------------------------------------
0       4     bytes   Magic: b'NEC\\x02'
4       2     uint16  Width
6       2     uint16  Height
8       1     uint8   Time bins (B)
9       4     float32 Threshold C
13      4     uint32  Number of frames
17      4     float32 Frames per second
21      ...   frames  One record per frame:
                        uint32  compressed length L
                        L bytes zlib stream of:
                          uint32  n  (number of non-zero cells kept)
                          4*n     byte planes of the uint32 index deltas
                                  (plane 0 = least significant bytes, ...)
                          n       int8 values  (value = int8 / 127)

Indices are flat positions in the [B, H, W] voxel, sorted ascending, stored as
deltas from the previous index.  Splitting the deltas into byte planes makes
the high planes almost all zeros, which DEFLATE compresses to nothing.

Version 1 (magic ``NEC\\x01``, raw int32 indices + float32 values, 13-byte
header, no fps) is still readable.
"""

from __future__ import annotations

import logging
import math
import struct
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterator, List, Optional, Tuple

import numpy as np

from neurocodec.simulator import DVSSimulator, EventStream, FrameEvents

logger = logging.getLogger(__name__)

MAGIC_V1 = b"NEC\x01"
MAGIC = b"NEC\x02"
_HEADER_V1 = struct.Struct("<HHBfI")
_HEADER_V2 = struct.Struct("<HHBfIf")
_QUANT = 127.0


# ---------------------------------------------------------------------------
# Voxel grid builder
# ---------------------------------------------------------------------------

class VoxelGrid:
    """
    Converts events into voxel grids ``[B x H x W]``, one per video frame.

    Bin ``k`` of the voxel for frame ``t`` holds the summed polarity of the
    events of frame ``t - (B - 1) + k``.  The result is L-infinity-normalised
    so values lie in [-1, 1].
    """

    def __init__(self, n_bins: int = 5) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.n_bins = n_bins

    def window_voxel(self, window: List[Optional[FrameEvents]], height: int, width: int) -> np.ndarray:
        """
        Build one voxel from the last ``B`` frames of events (oldest first).
        ``window`` may be shorter than ``B`` at the start of a clip, or
        contain ``None`` for empty frames.
        """
        B = self.n_bins
        voxel = np.zeros((B, height, width), dtype=np.float32)
        offset = B - len(window)
        for k, ev in enumerate(window):
            if ev is None or len(ev[0]) == 0:
                continue
            xs, ys, ps = ev
            np.add.at(voxel[offset + k], (ys, xs), ps.astype(np.float32))
        mx = float(np.abs(voxel).max())
        if mx > 0:
            voxel /= mx
        return voxel

    def iter_frames(self, per_frame_events: Iterator[FrameEvents], height: int, width: int
                    ) -> Iterator[np.ndarray]:
        """Streaming build: one ``[B,H,W]`` voxel per input frame."""
        window: Deque[FrameEvents] = deque(maxlen=self.n_bins)
        for ev in per_frame_events:
            window.append(ev)
            yield self.window_voxel(list(window), height, width)

    def build(self, stream: EventStream) -> np.ndarray:
        """
        Dense build for small clips / training.

        Returns
        -------
        np.ndarray  shape (n_frames, B, H, W), dtype float32, values in [-1, 1]
        """
        B, H, W, T = self.n_bins, stream.height, stream.width, stream.n_frames
        if T == 0 or H == 0 or W == 0:
            return np.zeros((0, B, H, W), dtype=np.float32)
        frames = (stream.frame_events(t) for t in range(T))
        return np.stack(list(self.iter_frames(frames, H, W)))


# ---------------------------------------------------------------------------
# Top-K sparsification
# ---------------------------------------------------------------------------

def topk_sparsify(
    voxel: np.ndarray, k_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the ``k_frac`` fraction of cells with the largest absolute value.

    ``k_frac`` is taken relative to the number of *non-zero* cells when the
    voxel has any (so a sparse event voxel is not padded with zeros), with a
    minimum of one cell.

    Returns
    -------
    indices : np.ndarray  int32   flat indices, sorted ascending
    values  : np.ndarray  float32 corresponding values
    """
    if not 0 < k_frac <= 1:
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    flat = voxel.ravel()
    nz   = np.flatnonzero(flat)
    if nz.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    k = min(nz.size, max(1, math.ceil(nz.size * k_frac)))
    if k < nz.size:
        keep = np.argpartition(np.abs(flat[nz]), -k)[-k:]
        nz = nz[keep]
    nz = np.sort(nz)
    return nz.astype(np.int32), flat[nz].astype(np.float32)


# ---------------------------------------------------------------------------
# Frame (de)serialisation
# ---------------------------------------------------------------------------

def _pack_frame(indices: np.ndarray, values: np.ndarray) -> bytes:
    n = len(indices)
    deltas = np.diff(indices.astype(np.int64), prepend=0).astype(np.uint32)
    planes = deltas.view(np.uint8).reshape(n, 4).T  # (4, n): byte planes
    q = np.clip(np.rint(values * _QUANT), -_QUANT, _QUANT).astype(np.int8)
    raw = struct.pack("<I", n) + planes.tobytes() + q.tobytes()
    return zlib.compress(raw, 6)


def _unpack_frame(blob: bytes, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        raw = zlib.decompress(blob)
        (n,) = struct.unpack_from("<I", raw, 0)
        if len(raw) != 4 + 5 * n:
            raise ValueError("frame payload length mismatch")
    except (zlib.error, struct.error) as e:
        raise ValueError(f"Corrupt .nec frame: {e}") from e
    planes = np.frombuffer(raw, dtype=np.uint8, count=4 * n, offset=4).reshape(4, n)
    deltas = np.ascontiguousarray(planes.T).view(np.uint32).reshape(n)
    indices = np.cumsum(deltas.astype(np.int64))
    if n and indices[-1] >= n_cells:
        raise ValueError("Corrupt .nec frame: index out of range")
    q = np.frombuffer(raw, dtype=np.int8, count=n, offset=4 + 4 * n)
    return indices.astype(np.int32), q.astype(np.float32) / _QUANT


# ---------------------------------------------------------------------------
# .nec writer / reader
# ---------------------------------------------------------------------------

@dataclass
class NECFrame:
    indices: np.ndarray  # int32 flat indices
    values:  np.ndarray  # float32 values


class NECWriter:
    """Write a sequence of voxel grids to a .nec (v2) file."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        n_bins: int,
        threshold: float,
        n_frames: int = 0,
        k_frac: float = 0.5,
        fps: float = 30.0,
    ) -> None:
        self.path      = Path(path)
        self.width     = width
        self.height    = height
        self.n_bins    = n_bins
        self.threshold = threshold
        self.n_frames  = n_frames   # informational; the real count is what was added
        self.k_frac    = k_frac
        self.fps       = fps
        self._blobs: List[bytes] = []

    def add_voxel(self, voxel: np.ndarray) -> None:
        """Add a single [B x H x W] voxel grid."""
        if voxel.shape != (self.n_bins, self.height, self.width):
            raise ValueError(
                f"voxel shape {voxel.shape} != {(self.n_bins, self.height, self.width)}")
        idx, vals = topk_sparsify(voxel, self.k_frac)
        self._blobs.append(_pack_frame(idx, vals))

    def write(self) -> int:
        """Flush to disk.  Returns file size in bytes."""
        n = len(self._blobs)
        if self.n_frames and self.n_frames != n:
            logger.warning("Declared %d frames but %d were added; writing %d.",
                           self.n_frames, n, n)
        with open(self.path, "wb") as f:
            f.write(MAGIC)
            f.write(_HEADER_V2.pack(self.width, self.height, self.n_bins,
                                    self.threshold, n, self.fps))
            for blob in self._blobs:
                f.write(struct.pack("<I", len(blob)))
                f.write(blob)

        size = self.path.stat().st_size
        logger.info("Wrote %s (%.2f KB, %d frames)", self.path.name, size / 1024, n)
        return size


class NECReader:
    """Read a .nec file (v1 or v2) and iterate over sparse voxel frames."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic == MAGIC:
                self.version = 2
                hdr = _HEADER_V2
            elif magic == MAGIC_V1:
                self.version = 1
                hdr = _HEADER_V1
            else:
                raise ValueError(f"Not a .nec file: {self.path}")
            raw = f.read(hdr.size)
            if len(raw) != hdr.size:
                raise ValueError(f"Truncated .nec header: {self.path}")
            fields = hdr.unpack(raw)
            self.width, self.height, self.n_bins, self.threshold, self.n_frames = fields[:5]
            self.fps = float(fields[5]) if self.version == 2 else 30.0
            self._data_offset = f.tell()
        if min(self.width, self.height, self.n_bins) < 1:
            raise ValueError(f"Invalid .nec header (zero dimension): {self.path}")

    @property
    def n_cells(self) -> int:
        return self.n_bins * self.height * self.width

    def _read_exact(self, f, n: int) -> bytes:
        data = f.read(n)
        if len(data) != n:
            raise ValueError(f"Truncated .nec file: {self.path}")
        return data

    def iter_frames(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(indices int32, values float32)`` per frame."""
        with open(self.path, "rb") as f:
            f.seek(self._data_offset)
            for _ in range(self.n_frames):
                (n,) = struct.unpack("<I", self._read_exact(f, 4))
                if self.version == 2:
                    yield _unpack_frame(self._read_exact(f, n), self.n_cells)
                else:
                    indices = np.frombuffer(self._read_exact(f, 4 * n), dtype=np.int32).copy()
                    values  = np.frombuffer(self._read_exact(f, 4 * n), dtype=np.float32).copy()
                    if n and (indices.min() < 0 or indices.max() >= self.n_cells):
                        raise ValueError("Corrupt .nec frame: index out of range")
                    yield indices, values

    def to_voxel(self, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Scatter one sparse frame into a dense [B, H, W] voxel."""
        flat = np.zeros(self.n_cells, dtype=np.float32)
        flat[indices] = values
        return flat.reshape(self.n_bins, self.height, self.width)

    def iter_dense(self) -> Iterator[np.ndarray]:
        """Yield dense [B, H, W] voxels one frame at a time (constant memory)."""
        for idx, vals in self.iter_frames():
            yield self.to_voxel(idx, vals)

    def to_dense(self) -> np.ndarray:
        """Reconstruct the full dense voxel tensor [n_frames, B, H, W] (needs lots of RAM)."""
        grids = np.zeros((self.n_frames, self.n_bins, self.height, self.width), dtype=np.float32)
        for t, voxel in enumerate(self.iter_dense()):
            grids[t] = voxel
        return grids


# ---------------------------------------------------------------------------
# Compression ratio helper
# ---------------------------------------------------------------------------

def compression_ratio(original_path: str | Path, nec_path: str | Path) -> float:
    """Original file size divided by .nec size (values < 1 mean the .nec is larger)."""
    orig_size = Path(original_path).stat().st_size
    nec_size  = Path(nec_path).stat().st_size
    return orig_size / max(1, nec_size)


# ---------------------------------------------------------------------------
# VoxelEncoder - high-level class
# ---------------------------------------------------------------------------

class VoxelEncoder:
    """
    High-level encoder that wraps DVS simulation + voxel building + .nec writing.
    Frames are streamed, so memory use does not grow with clip length.

    Example
    -------
    >>> enc = VoxelEncoder(threshold=0.2, n_bins=5, k_frac=0.5)
    >>> enc.encode("dashcam.mp4", "dashcam.nec")
    """

    def __init__(
        self,
        threshold: float = 0.2,
        n_bins: int = 5,
        k_frac: float = 0.5,
    ) -> None:
        if not 0 < k_frac <= 1:
            raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
        if not 1 <= n_bins <= 255:
            raise ValueError(f"n_bins must be in [1, 255], got {n_bins}")
        self.threshold = threshold
        self.n_bins    = n_bins
        self.k_frac    = k_frac

    def encode(self, video_path: str | Path, nec_path: str | Path) -> dict:
        """
        Encode a video to .nec format.

        Returns dict with stats: n_events, n_frames, event_density,
        file_size_kb, compression_ratio (original file size / .nec size).
        """
        import cv2

        video_path = Path(video_path)
        nec_path   = Path(nec_path)

        sim = DVSSimulator(threshold=self.threshold)
        vg  = VoxelGrid(n_bins=self.n_bins)

        logger.info("Encoding %s (C=%.2f, B=%d, K=%.2f)...",
                    video_path.name, self.threshold, self.n_bins, self.k_frac)
        cap = sim.open_video(video_path)
        try:
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = NECWriter(nec_path, width, height, self.n_bins, self.threshold,
                               k_frac=self.k_frac, fps=fps)

            n_events = 0

            def counted() -> Iterator[FrameEvents]:
                nonlocal n_events
                for ev in sim.iter_events(sim.iter_video_frames(cap)):
                    n_events += len(ev[0])
                    yield ev

            for voxel in vg.iter_frames(counted(), height, width):
                writer.add_voxel(voxel)
        finally:
            cap.release()

        n_frames = len(writer._blobs)
        if n_frames == 0:
            raise RuntimeError("Could not read first frame.")
        file_size = writer.write()

        ratio = compression_ratio(video_path, nec_path)
        stats = {
            "n_events":          n_events,
            "n_frames":          n_frames,
            "event_density":     n_events / (n_frames * width * height),
            "file_size_kb":      file_size / 1024,
            "compression_ratio": ratio,
        }
        logger.info(
            "Encoded -> %s | ratio=%.2fx | %.1f KB",
            nec_path.name, ratio, file_size / 1024,
        )
        return stats


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def encode(
    video_path: str | Path,
    nec_path: str | Path,
    threshold: float = 0.2,
    n_bins: int = 5,
    k_frac: float = 0.5,
) -> dict:
    """
    Encode a video to .nec format.

    Parameters
    ----------
    video_path : str | Path  Input video (any format OpenCV can read).
    nec_path   : str | Path  Output .nec file.
    threshold  : float       DVS threshold C (default 0.2).
    n_bins     : int         Voxel time bins (default 5).
    k_frac     : float       Fraction of non-zero voxel cells to keep (default 0.5).

    Returns
    -------
    dict  Encoding statistics.
    """
    enc = VoxelEncoder(threshold=threshold, n_bins=n_bins, k_frac=k_frac)
    return enc.encode(video_path, nec_path)
