# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
Voxel-Grid Entropy Encoder  →  .nec binary format
===================================================
Pipeline:
  EventStream  →  Voxel grid [B×H×W]  →  Top-K sparsification
               →  Huffman entropy coding  →  .nec binary file

.nec file format (NeuroCodec Event Container)
---------------------------------------------
Offset  Size  Type    Field
------  ----  ------  -----------------------------------------
0       4     uint8   Magic: b'NEC\x01'
4       2     uint16  Width
6       2     uint16  Height
8       1     uint8   Time bins (B)
9       4     float32 Threshold C
13      4     uint32  Number of frames
17      4     uint32  Number of sparse (index, value) pairs
21      4*N   int32   Sparse flat indices (B*H*W space)
21+4N   4*N   float32 Sparse values
"""

from __future__ import annotations

import heapq
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurocodec.simulator import DVSSimulator, EventStream, simulate

logger = logging.getLogger(__name__)

MAGIC = b"NEC\x01"


# ---------------------------------------------------------------------------
# Voxel grid builder
# ---------------------------------------------------------------------------

class VoxelGrid:
    """
    Converts an EventStream into a dense 3-D tensor [B × H × W].

    Time is divided into B equal bins; events are accumulated (summed)
    into the appropriate spatial cell.  The result is L∞-normalised so
    values lie in [-1, 1].
    """

    def __init__(self, n_bins: int = 5) -> None:
        self.n_bins = n_bins

    def build(self, stream: EventStream) -> np.ndarray:
        """
        Parameters
        ----------
        stream : EventStream

        Returns
        -------
        np.ndarray  shape (n_frames, B, H, W), dtype float32, values in [-1,1]
        """
        B, H, W, T = self.n_bins, stream.height, stream.width, stream.n_frames
        if T == 0 or H == 0 or W == 0:
            return np.zeros((0, B, H, W), dtype=np.float32)

        # One voxel grid per frame (frame index = event.t)
        grids = np.zeros((T, B, H, W), dtype=np.float32)

        events_arr = stream.to_numpy()  # (N, 4): x y t p
        if events_arr.shape[0] == 0:
            return grids

        xs  = events_arr[:, 0]
        ys  = events_arr[:, 1]
        ts  = events_arr[:, 2]           # frame indices
        ps  = events_arr[:, 3].astype(np.float32)

        # Map frame index → time bin
        bin_idx = np.clip((ts * B) // T, 0, B - 1)

        np.add.at(grids, (ts, bin_idx, ys, xs), ps)

        # L-inf normalise per frame
        for f in range(T):
            mx = np.abs(grids[f]).max()
            if mx > 0:
                grids[f] /= mx

        return grids


# ---------------------------------------------------------------------------
# Huffman coder (pure Python, no external deps)
# ---------------------------------------------------------------------------

class HuffmanCoder:
    """Simple Huffman encoder / decoder for integer symbol streams."""

    @staticmethod
    def build_tree(symbols: List[int]) -> Dict[int, str]:
        if not symbols:
            return {}
        freq: Dict[int, int] = {}
        for s in symbols:
            freq[s] = freq.get(s, 0) + 1

        # heap: (freq, counter, node)
        heap: list = []
        counter = 0
        for sym, f in freq.items():
            heapq.heappush(heap, (f, counter, sym))
            counter += 1

        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap)
            f2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (f1 + f2, counter, (n1, n2)))
            counter += 1

        _, _, root = heap[0]
        codes: Dict[int, str] = {}
        HuffmanCoder._traverse(root, "", codes)
        return codes

    @staticmethod
    def _traverse(node, prefix: str, codes: Dict[int, str]) -> None:
        if isinstance(node, tuple):
            HuffmanCoder._traverse(node[0], prefix + "0", codes)
            HuffmanCoder._traverse(node[1], prefix + "1", codes)
        else:
            codes[node] = prefix if prefix else "0"

    @staticmethod
    def encode(symbols: List[int], codes: Dict[int, str]) -> bytes:
        """Pack symbols into a byte string using the given codebook."""
        bits = "".join(codes[s] for s in symbols)
        # Pad to byte boundary
        pad = (8 - len(bits) % 8) % 8
        bits += "0" * pad
        data = bytearray()
        data.append(pad)  # store padding length
        for i in range(0, len(bits), 8):
            data.append(int(bits[i : i + 8], 2))
        return bytes(data)

    @staticmethod
    def decode(data: bytes, codes: Dict[int, str], n_symbols: int) -> List[int]:
        """Decode byte string back to symbols."""
        pad      = data[0]
        bits     = "".join(f"{b:08b}" for b in data[1:])
        if pad:
            bits = bits[:-pad]
        inv      = {v: k for k, v in codes.items()}
        symbols  = []
        buf      = ""
        for bit in bits:
            buf += bit
            if buf in inv:
                symbols.append(inv[buf])
                buf = ""
                if len(symbols) == n_symbols:
                    break
        return symbols


# ---------------------------------------------------------------------------
# Top-K sparsification
# ---------------------------------------------------------------------------

def topk_sparsify(
    voxel: np.ndarray, k_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only the top-K fraction of values by absolute magnitude.

    Returns
    -------
    indices : np.ndarray  int32   flat indices into the voxel
    values  : np.ndarray  float32 corresponding values
    """
    flat   = voxel.ravel()
    k      = max(1, int(len(flat) * k_frac))
    top_k  = np.argpartition(np.abs(flat), -k)[-k:]
    top_k  = top_k[np.argsort(np.abs(flat[top_k]))[::-1]]  # sort descending
    return top_k.astype(np.int32), flat[top_k].astype(np.float32)


# ---------------------------------------------------------------------------
# .nec writer / reader
# ---------------------------------------------------------------------------

@dataclass
class NECFrame:
    indices: np.ndarray  # int32 flat indices
    values:  np.ndarray  # float32 values


class NECWriter:
    """Write a sequence of voxel grids to a .nec file."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        n_bins: int,
        threshold: float,
        n_frames: int,
        k_frac: float = 0.1,
    ) -> None:
        self.path     = Path(path)
        self.width    = width
        self.height   = height
        self.n_bins   = n_bins
        self.threshold = threshold
        self.n_frames = n_frames
        self.k_frac   = k_frac
        self._frames: List[NECFrame] = []

    def add_voxel(self, voxel: np.ndarray) -> None:
        """Add a single [B×H×W] voxel grid."""
        idx, vals = topk_sparsify(voxel, self.k_frac)
        self._frames.append(NECFrame(idx, vals))

    def write(self) -> int:
        """Flush to disk.  Returns file size in bytes."""
        with open(self.path, "wb") as f:
            # Header
            f.write(MAGIC)
            f.write(struct.pack("<HHBfI",
                self.width, self.height, self.n_bins,
                self.threshold, self.n_frames))

            for frame in self._frames:
                n = len(frame.indices)
                f.write(struct.pack("<I", n))
                f.write(frame.indices.tobytes())
                f.write(frame.values.tobytes())

        size = self.path.stat().st_size
        logger.info("Wrote %s (%.2f KB, %d frames)", self.path.name, size / 1024, self.n_frames)
        return size


class NECReader:
    """Read a .nec file and iterate over sparse voxel frames."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Not a .nec file: {self.path}")
            self.width, self.height, self.n_bins, self.threshold, self.n_frames = \
                struct.unpack("<HHBfI", f.read(13))
            self._data_offset = f.tell()

    def iter_frames(self) -> "Iterator[Tuple[np.ndarray, np.ndarray]]":
        with open(self.path, "rb") as f:
            f.seek(self._data_offset)
            for _ in range(self.n_frames):
                (n,) = struct.unpack("<I", f.read(4))
                indices = np.frombuffer(f.read(4 * n), dtype=np.int32).copy()
                values  = np.frombuffer(f.read(4 * n), dtype=np.float32).copy()
                yield indices, values

    def to_dense(self) -> np.ndarray:
        """Reconstruct full dense voxel tensor [n_frames, B, H, W]."""
        B, H, W = self.n_bins, self.height, self.width
        grids = np.zeros((self.n_frames, B, H, W), dtype=np.float32)
        for t, (idx, vals) in enumerate(self.iter_frames()):
            flat = np.zeros(B * H * W, dtype=np.float32)
            flat[idx] = vals
            grids[t] = flat.reshape(B, H, W)
        return grids


# ---------------------------------------------------------------------------
# Compression ratio helper
# ---------------------------------------------------------------------------

def compression_ratio(original_path: str | Path, nec_path: str | Path) -> float:
    orig_size = Path(original_path).stat().st_size
    nec_size  = Path(nec_path).stat().st_size
    return orig_size / max(1, nec_size)


# ---------------------------------------------------------------------------
# VoxelEncoder — high-level class
# ---------------------------------------------------------------------------

class VoxelEncoder:
    """
    High-level encoder that wraps DVS simulation + voxel building + .nec writing.

    Example
    -------
    >>> enc = VoxelEncoder(threshold=0.2, n_bins=5, k_frac=0.1)
    >>> enc.encode("dashcam.mp4", "dashcam.nec")
    """

    def __init__(
        self,
        threshold: float = 0.2,
        n_bins: int = 5,
        k_frac: float = 0.1,
    ) -> None:
        self.threshold = threshold
        self.n_bins    = n_bins
        self.k_frac    = k_frac

    def encode(self, video_path: str | Path, nec_path: str | Path) -> dict:
        """
        Encode a video to .nec format.

        Returns dict with stats: n_events, n_frames, file_size_kb,
        compression_ratio (vs raw video).
        """
        video_path = Path(video_path)
        nec_path   = Path(nec_path)

        logger.info("Simulating events from %s (C=%.2f)...", video_path.name, self.threshold)
        sim    = DVSSimulator(threshold=self.threshold)
        stream = sim.simulate_video(video_path)

        logger.info("Building voxel grids (B=%d)...", self.n_bins)
        vg     = VoxelGrid(n_bins=self.n_bins)
        grids  = vg.build(stream)

        writer = NECWriter(
            nec_path,
            width=stream.width,
            height=stream.height,
            n_bins=self.n_bins,
            threshold=self.threshold,
            n_frames=stream.n_frames,
            k_frac=self.k_frac,
        )
        for t in range(grids.shape[0]):
            writer.add_voxel(grids[t])
        file_size = writer.write()

        ratio = compression_ratio(video_path, nec_path)
        stats = {
            "n_events":          len(stream),
            "n_frames":          stream.n_frames,
            "event_density":     stream.density(),
            "file_size_kb":      file_size / 1024,
            "compression_ratio": ratio,
        }
        logger.info(
            "Encoded → %s | ratio=%.1fx | %.1f KB",
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
    k_frac: float = 0.1,
) -> dict:
    """
    Encode a video to .nec format.

    Parameters
    ----------
    video_path : str | Path  Input video (any format OpenCV can read).
    nec_path   : str | Path  Output .nec file.
    threshold  : float       DVS threshold C (default 0.2).
    n_bins     : int         Voxel time bins (default 5).
    k_frac     : float       Top-K sparsity fraction (default 0.1 = 10%).

    Returns
    -------
    dict  Encoding statistics.
    """
    enc = VoxelEncoder(threshold=threshold, n_bins=n_bins, k_frac=k_frac)
    return enc.encode(video_path, nec_path)
