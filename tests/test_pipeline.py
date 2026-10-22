# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""End-to-end tests: video -> .nec -> video, and the CLI."""

import cv2
import numpy as np
import pytest
import torch

from neurocodec import encode, decode, simulate
from neurocodec.cli import main
from neurocodec.decoder import (
    ConvLSTMDecoder, DecoderStep, export_onnx, load_decoder,
)
from neurocodec.encoder import NECReader


@pytest.fixture
def video(tmp_path):
    """Small synthetic clip: a white square sliding over a noisy-free dark background."""
    path = tmp_path / "clip.mp4"
    H, W, T = 48, 64, 12
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (W, H))
    for i in range(T):
        f = np.full((H, W, 3), 30, np.uint8)
        f[10:30, 4 + 4 * i: 24 + 4 * i] = 220
        w.write(f)
    w.release()
    return path


class TestEncodeDecode:
    def test_simulate_video(self, video):
        stream = simulate(video, threshold=0.2)
        assert stream.n_frames == 12 and stream.width == 64 and stream.height == 48
        assert len(stream) > 0
        assert np.all(np.diff(stream.t) >= 0)

    def test_encode_stats_and_reader(self, video, tmp_path):
        out = tmp_path / "c.nec"
        stats = encode(video, out, threshold=0.2, n_bins=4, k_frac=1.0)
        r = NECReader(out)
        assert (r.n_frames, r.n_bins, r.width, r.height) == (12, 4, 64, 48)
        assert r.fps == pytest.approx(15.0)
        assert stats["n_frames"] == 12 and stats["n_events"] > 0
        assert stats["file_size_kb"] > 0 and stats["compression_ratio"] > 0

    def test_encode_matches_dense_voxel_build(self, video, tmp_path):
        from neurocodec.encoder import VoxelGrid
        out = tmp_path / "c.nec"
        encode(video, out, threshold=0.2, n_bins=3, k_frac=1.0)
        expected = VoxelGrid(3).build(simulate(video, threshold=0.2))
        np.testing.assert_allclose(NECReader(out).to_dense(), expected, atol=0.5 / 127 + 1e-6)

    def test_decode_requires_weights(self, video, tmp_path):
        nec = tmp_path / "c.nec"
        encode(video, nec)
        with pytest.raises(FileNotFoundError):
            decode(nec, tmp_path / "o.mp4", weights_path=tmp_path / "missing.pt")

    def test_decode_with_checkpoint(self, video, tmp_path):
        nec = tmp_path / "c.nec"
        encode(video, nec, n_bins=5)
        ckpt = tmp_path / "w.pt"
        model = ConvLSTMDecoder(n_bins=5, hidden_dims=[8, 8, 4])
        torch.save({"model_state_dict": model.state_dict(),
                    "config": {"hidden_dims": [8, 8, 4]}}, ckpt)
        res = decode(nec, tmp_path / "o.mp4", weights_path=ckpt, device="cpu")
        assert res["n_frames"] == 12 and res["fps"] == pytest.approx(15.0)
        cap = cv2.VideoCapture(str(tmp_path / "o.mp4"))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12

    def test_decode_allow_random(self, video, tmp_path):
        nec = tmp_path / "c.nec"
        encode(video, nec)
        res = decode(nec, tmp_path / "o.mp4", device="cpu", allow_random_weights=True)
        assert res["n_frames"] == 12

    def test_strict_weight_mismatch_raises(self, tmp_path):
        ckpt = tmp_path / "w.pt"
        torch.save(ConvLSTMDecoder(n_bins=3).state_dict(), ckpt)
        with pytest.raises(RuntimeError):
            load_decoder(ckpt, n_bins=5, device=torch.device("cpu"))

    def test_missing_video(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            encode(tmp_path / "nope.mp4", tmp_path / "x.nec")


class TestStatefulDecoder:
    def test_gradients_flow_across_timesteps(self):
        """The state must not be detached between frames (needed for BPTT)."""
        torch.manual_seed(0)
        model = ConvLSTMDecoder(n_bins=2, hidden_dims=[4, 4])
        x0 = torch.randn(1, 2, 8, 8, requires_grad=True)
        model(x0)
        out = model(torch.zeros(1, 2, 8, 8))
        out.sum().backward()
        assert x0.grad is not None and x0.grad.abs().sum() > 0

    def test_detach_state(self):
        model = ConvLSTMDecoder(n_bins=2, hidden_dims=[4, 4])
        model(torch.randn(1, 2, 8, 8))
        model.detach_state()
        assert all(not h.requires_grad and not c.requires_grad for h, c in model._hidden)

    def test_step_wrapper_matches_stateful_forward(self):
        torch.manual_seed(0)
        model = ConvLSTMDecoder(n_bins=3, hidden_dims=[6, 6, 4]).eval()
        step = DecoderStep(model)
        H, W = 10, 12
        state = [torch.zeros(1, ch, H, W) for ch in step.state_channels for _ in (0, 1)]
        model.reset_state()
        with torch.no_grad():
            for _ in range(3):
                v = torch.randn(1, 3, H, W)
                ref = model(v)
                frame, *state = step(v, *state)
                torch.testing.assert_close(frame, ref, atol=1e-5, rtol=1e-4)

    def test_onnx_export_roundtrip(self, tmp_path):
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnx")
        model = ConvLSTMDecoder(n_bins=3, hidden_dims=[6, 4]).eval()
        path = tmp_path / "d.onnx"
        names = export_onnx(model, path, height=16, width=16)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        step = DecoderStep(model)
        H, W = 20, 24    # different from the export size: dynamic axes
        state = {n: np.zeros((1, ch, H, W), np.float32)
                 for i, ch in enumerate(step.state_channels) for n in (f"h{i}", f"c{i}")}
        out_names = [o.name for o in sess.get_outputs()]
        model.reset_state()
        rng = np.random.default_rng(0)
        for _ in range(3):
            v = rng.standard_normal((1, 3, H, W)).astype(np.float32)
            with torch.no_grad():
                ref = model(torch.from_numpy(v)).numpy()
            outs = sess.run(out_names, {"voxel": v, **state})
            np.testing.assert_allclose(outs[0], ref, atol=1e-4)
            state = dict(zip(names[1:], outs[1:]))


class TestCLI:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as e:
            main(["--version"])
        assert e.value.code == 0

    def test_encode_info_simulate_benchmark(self, video, tmp_path, capsys):
        nec = tmp_path / "c.nec"
        assert main(["encode", str(video), str(nec)]) == 0
        assert "Encoding complete" in capsys.readouterr().out
        assert main(["info", str(nec)]) == 0
        assert "12" in capsys.readouterr().out
        ev = tmp_path / "ev.mp4"
        assert main(["simulate", str(video), "--output", str(ev)]) == 0
        assert ev.exists()
        js = tmp_path / "b.json"
        assert main(["benchmark", str(video), "--thresholds", "0.1,0.3",
                     "--output", str(js)]) == 0
        assert js.exists()

    def test_decode_without_weights_fails_cleanly(self, video, tmp_path, capsys):
        nec = tmp_path / "c.nec"
        main(["encode", str(video), str(nec)])
        capsys.readouterr()
        assert main(["decode", str(nec), str(tmp_path / "o.mp4"),
                     "--weights", str(tmp_path / "missing.pt")]) == 1
        assert "error:" in capsys.readouterr().err

    def test_decode_allow_random(self, video, tmp_path, capsys):
        nec = tmp_path / "c.nec"
        main(["encode", str(video), str(nec)])
        assert main(["decode", str(nec), str(tmp_path / "o.mp4"),
                     "--allow-random", "--device", "cpu"]) == 0

    def test_missing_input(self, tmp_path, capsys):
        assert main(["info", str(tmp_path / "nope.nec")]) == 1
