# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""Tests for voxel encoder and .nec format."""

import struct

import numpy as np
import pytest

from neurocodec.encoder import (
    MAGIC_V1, VoxelGrid, NECWriter, NECReader, topk_sparsify,
)
from neurocodec.simulator import DVSSimulator, EventStream
from tests.conftest import make_moving_frames


class TestVoxelGrid:
    def test_output_shape(self):
        stream = DVSSimulator(threshold=0.1).simulate_array(make_moving_frames())
        grids  = VoxelGrid(n_bins=5).build(stream)
        assert grids.shape == (stream.n_frames, 5, stream.height, stream.width)

    def test_values_in_minus1_plus1(self):
        stream = DVSSimulator(threshold=0.1).simulate_array(make_moving_frames())
        grids  = VoxelGrid(5).build(stream)
        assert grids.min() >= -1.0 - 1e-6
        assert grids.max() <=  1.0 + 1e-6

    def test_static_stream_all_zeros(self):
        stream = EventStream(width=16, height=16, n_frames=5, threshold=0.2)
        assert np.all(VoxelGrid(3).build(stream) == 0)

    def test_moving_stream_nonzero(self):
        stream = DVSSimulator(threshold=0.05).simulate_array(make_moving_frames())
        assert VoxelGrid(5).build(stream).max() > 0

    def test_bins_hold_distinct_frames(self):
        """Bin k of voxel t must hold the events of frame t-(B-1)+k (real temporal bins)."""
        B, H, W = 3, 4, 4
        stream = EventStream(
            x=np.array([0, 1, 2], np.int32), y=np.array([0, 1, 2], np.int32),
            t=np.array([1, 2, 3], np.int32), p=np.array([1, -1, 1], np.int8),
            width=W, height=H, n_frames=5)
        grids = VoxelGrid(B).build(stream)
        # voxel for frame 3: bins = frames 1, 2, 3
        assert grids[3, 0, 0, 0] == 1 and grids[3, 1, 1, 1] == -1 and grids[3, 2, 2, 2] == 1
        # voxel for frame 1: only the last bin (current frame) is populated
        assert grids[1, 2, 0, 0] == 1 and np.count_nonzero(grids[1]) == 1
        # more than one bin is active in a single voxel
        assert (np.abs(grids[3]).reshape(B, -1).max(axis=1) > 0).sum() == 3

    def test_invalid_bins(self):
        with pytest.raises(ValueError):
            VoxelGrid(0)


class TestTopKSparsify:
    def test_keeps_k_elements(self):
        arr = np.random.randn(100).astype(np.float32)
        idx, vals = topk_sparsify(arr.reshape(10, 10), k_frac=0.2)
        assert len(idx) == 20

    def test_values_match_original(self):
        arr = np.arange(100, dtype=np.float32)
        idx, vals = topk_sparsify(arr.reshape(10, 10), k_frac=0.1)
        np.testing.assert_allclose(arr.ravel()[idx], vals)

    def test_keeps_largest_magnitudes(self):
        arr = np.array([0.1, -0.9, 0.0, 0.5, -0.2, 0.0], np.float32)
        idx, vals = topk_sparsify(arr, k_frac=0.5)   # 4 non-zero -> keep 2
        assert set(idx.tolist()) == {1, 3}

    def test_indices_sorted(self):
        idx, _ = topk_sparsify(np.random.randn(200).astype(np.float32), 0.3)
        assert np.all(np.diff(idx) > 0)

    def test_minimum_one_element(self):
        idx, _ = topk_sparsify(np.ones((2, 2), dtype=np.float32), k_frac=0.001)
        assert len(idx) == 1

    def test_all_zero_voxel_gives_nothing(self):
        idx, vals = topk_sparsify(np.zeros((3, 4, 4), np.float32), 0.5)
        assert len(idx) == 0 and len(vals) == 0

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            topk_sparsify(np.ones(4, np.float32), 0.0)


def _random_voxel(B, H, W, density=0.1, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros((B, H, W), np.float32)
    mask = rng.random(v.shape) < density
    v[mask] = rng.choice([-1.0, 1.0], size=int(mask.sum()))
    return v


class TestNECFormat:
    def test_write_read_roundtrip(self, tmp_path):
        path = tmp_path / "a.nec"
        W, H, B, T = 16, 12, 5, 4
        writer = NECWriter(path, W, H, B, 0.2, T, k_frac=1.0, fps=24.0)
        voxels = [_random_voxel(B, H, W, seed=i) for i in range(T)]
        for v in voxels:
            writer.add_voxel(v)
        writer.write()

        reader = NECReader(path)
        assert (reader.width, reader.height, reader.n_bins, reader.n_frames) == (W, H, B, T)
        assert reader.fps == pytest.approx(24.0)
        assert reader.version == 2
        dense = reader.to_dense()
        assert dense.shape == (T, B, H, W)
        for t in range(T):   # +-1 values quantise exactly
            np.testing.assert_allclose(dense[t], voxels[t])

    def test_quantisation_error_bounded(self, tmp_path):
        path = tmp_path / "q.nec"
        v = np.zeros((2, 8, 8), np.float32)
        v[0, 1, 1], v[1, 2, 3] = 0.3337, -0.7771
        w = NECWriter(path, 8, 8, 2, 0.2, 1, k_frac=1.0)
        w.add_voxel(v)
        w.write()
        got = NECReader(path).to_dense()[0]
        assert np.abs(got - v).max() <= 0.5 / 127 + 1e-6

    def test_empty_frames_roundtrip(self, tmp_path):
        path = tmp_path / "e.nec"
        w = NECWriter(path, 8, 8, 3, 0.2, 2)
        for _ in range(2):
            w.add_voxel(np.zeros((3, 8, 8), np.float32))
        w.write()
        assert not NECReader(path).to_dense().any()

    def test_large_index_gaps(self, tmp_path):
        """Deltas above 16 bits must survive the byte-plane coding."""
        path = tmp_path / "big.nec"
        H, W, B = 400, 400, 2
        v = np.zeros((B, H, W), np.float32)
        v.ravel()[[0, 70_000, 250_000, B * H * W - 1]] = [1, -1, 1, -1]
        w = NECWriter(path, W, H, B, 0.2, 1, k_frac=1.0)
        w.add_voxel(v)
        w.write()
        np.testing.assert_array_equal(NECReader(path).to_dense()[0], v)

    def test_compresses_sparse_data(self, tmp_path):
        path = tmp_path / "c.nec"
        B, H, W = 5, 120, 160
        w = NECWriter(path, W, H, B, 0.2, 1, k_frac=1.0)
        w.add_voxel(_random_voxel(B, H, W, density=0.02))
        size = w.write()
        n = int(0.02 * B * H * W)
        assert size < 8 * n   # old raw format cost 8 bytes per kept cell

    def test_magic_bytes_validated(self, tmp_path):
        path = tmp_path / "bad.nec"
        path.write_bytes(b"BADMAGIC")
        with pytest.raises(ValueError):
            NECReader(path)

    def test_truncated_header(self, tmp_path):
        path = tmp_path / "t.nec"
        path.write_bytes(b"NEC\x02\x01\x00")
        with pytest.raises(ValueError):
            NECReader(path)

    def test_truncated_body(self, tmp_path):
        path = tmp_path / "t2.nec"
        w = NECWriter(path, 8, 8, 3, 0.2, 2)
        for _ in range(2):
            w.add_voxel(_random_voxel(3, 8, 8, 0.3))
        w.write()
        data = path.read_bytes()
        path.write_bytes(data[:-5])
        with pytest.raises(ValueError):
            list(NECReader(path).iter_frames())

    def test_shape_mismatch_rejected(self, tmp_path):
        w = NECWriter(tmp_path / "x.nec", 8, 8, 3, 0.2, 1)
        with pytest.raises(ValueError):
            w.add_voxel(np.zeros((3, 9, 8), np.float32))

    def test_file_size_nonzero(self, tmp_path):
        w = NECWriter(tmp_path / "s.nec", 8, 8, 3, 0.2, 2, k_frac=0.5)
        for _ in range(2):
            w.add_voxel(np.random.randn(3, 8, 8).astype(np.float32))
        assert w.write() > 0

    def test_iter_frames_count(self, tmp_path):
        path = tmp_path / "n.nec"
        T = 6
        w = NECWriter(path, 8, 8, 3, 0.2, T, k_frac=0.3)
        for _ in range(T):
            w.add_voxel(np.random.randn(3, 8, 8).astype(np.float32))
        w.write()
        assert sum(1 for _ in NECReader(path).iter_frames()) == T

    def test_reads_legacy_v1(self, tmp_path):
        path = tmp_path / "v1.nec"
        idx = np.array([1, 5], np.int32)
        vals = np.array([0.5, -1.0], np.float32)
        with open(path, "wb") as f:
            f.write(MAGIC_V1)
            f.write(struct.pack("<HHBfI", 4, 4, 1, 0.2, 1))
            f.write(struct.pack("<I", 2) + idx.tobytes() + vals.tobytes())
        r = NECReader(path)
        assert r.version == 1 and r.fps == 30.0
        dense = r.to_dense()
        assert dense[0, 0, 0, 1] == 0.5 and dense[0, 0, 1, 1] == -1.0
