# NeuroCodec 

**Neuromorphic-Inspired Spike-Driven Video Compression (research prototype)**

[![CI](https://github.com/chandini-narayana/neurocodec/actions/workflows/test.yml/badge.svg)](https://github.com/chandini-narayana/neurocodec/actions/workflows/test.yml)
[![Benchmark](https://github.com/chandini-narayana/neurocodec/actions/workflows/benchmark.yml/badge.svg)](https://github.com/chandini-narayana/neurocodec/actions/workflows/benchmark.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A software DVS event simulator + voxel-grid sparse encoder + ConvLSTM frame reconstructor.
> Works on any standard camera input — no event-camera hardware required.

---

## Status — please read

This is an **early research prototype**:

- ✅ DVS simulator, voxel encoder, `.nec` container, CLI, ONNX export and tests work.
- ⚠️ **No trained decoder weights are released yet.** `neurocodec decode` refuses to run
  without them (pass `--allow-random` only to test the pipeline — the output is a constant
  gray image).
- ⚠️ **No rate-distortion results have been measured yet.** Earlier drafts of this README
  and the paper listed SSIM/PSNR numbers vs H.264/H.265/AV1; those were placeholders, not
  measurements, and have been removed. `eval/benchmark_rd.py` produces real numbers once
  weights exist.
- ⚠️ **Compression is currently *not* competitive with H.264.** Measured on the 480×270,
  901-frame `sample.mp4` (1.5 MB H.264): `.nec` is 9.7 MB at `--kfrac 0.5` and 2.0 MB at
  `--kfrac 0.1`. Event data from natural video is dense; the research question is whether
  a learned decoder can make a very sparse event representation worthwhile.
- ⚠️ **Events only encode *changes*.** Static content (and absolute brightness) is not
  recoverable from events alone, so a decoder trained only on voxels will struggle there.
  Sending periodic key-frames is the obvious next step and is not implemented.

```
Input video → DVS simulator → Voxel grid [B×H×W] → Top-K → int8 + delta + DEFLATE → .nec → ConvLSTM decoder → grayscale video
```

---

## Installation

From source (the package is not published on PyPI yet):

```bash
git clone https://github.com/chandini-narayana/neurocodec
cd neurocodec
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. Optional extras: `demo` (Gradio),
`eval` (matplotlib), `onnx` (onnx + onnxruntime), `train` (jupyter), `release` (hatch, twine).

---

## Quick start

```bash
neurocodec simulate dashcam.mp4 --output events.mp4        # visualise DVS events
neurocodec encode   dashcam.mp4 dashcam.nec --threshold 0.2 --kfrac 0.5
neurocodec info     dashcam.nec
neurocodec decode   dashcam.nec out.mp4 --weights decoder_v1.pt   # needs trained weights
neurocodec benchmark dashcam.mp4 --thresholds 0.1,0.2,0.5   # events vs threshold
```

| Flag | Meaning |
|------|---------|
| `--threshold C` | log-intensity change that fires an event (lower → more events) |
| `--bins B` | temporal bins per voxel (bin *k* of frame *t* = events of frame *t−B+1+k*) |
| `--kfrac K` | fraction of the non-zero voxel cells kept (strongest first) |

### Python API

```python
from neurocodec import simulate, encode, decode

events = simulate("dashcam.mp4", threshold=0.2)
print(f"Events: {len(events):,} | events/pixel/frame: {events.density():.4f}")

stats = encode("dashcam.mp4", "dashcam.nec", threshold=0.2, k_frac=0.5)
print(f"{stats['file_size_kb']:.1f} KB | {stats['compression_ratio']:.2f}x vs input file")

decode("dashcam.nec", "reconstructed.mp4", weights_path="decoder_v1.pt")
```

### The `.nec` format (v2)

Header (21 bytes: magic `NEC\x02`, width, height, bins, threshold, frames, fps) followed by
one zlib-compressed record per frame. Each record stores the sorted indices of the kept voxel
cells as delta-coded byte planes plus int8-quantised values. Details in
[neurocodec/encoder.py](neurocodec/encoder.py). Version-1 files are still readable.

---

## Repository structure

```
neurocodec/        pip-installable package (simulator, encoder, convlstm, decoder, cli)
tests/             pytest suite (unit + end-to-end + CLI)
training/          Colab notebook that trains the decoder
eval/              rate-distortion benchmark, qualitative figures, ONNX export
demo/              Gradio app (HuggingFace Spaces)
cpp_runtime/       ONNX Runtime C++ decoder (stateful, reads .nec v1/v2)
paper/             LaTeX draft (results are placeholders — see the banner in the PDF)
docs/, assets/     training/publishing notes, model & dataset cards
```

---

## Training

Train the decoder on Google Colab (free T4):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chandini-narayana/neurocodec/blob/main/training/train_decoder.ipynb)

See [docs/TRAINING.md](docs/TRAINING.md). Put the resulting `decoder_v1.pt` in
`neurocodec/weights/` (git-ignored) or pass `--weights`.

## Benchmarking

```bash
pip install -e ".[eval]"
python eval/benchmark_rd.py --video dashcam.mp4 --weights decoder_v1.pt --output eval/results/
```

Needs `ffmpeg` (libx264/libx265/libaom-av1) on `PATH` for the baselines.

## C++ runtime

```bash
pip install -e ".[onnx]"
python eval/export_onnx.py --weights decoder_v1.pt --output decoder.onnx --height 270 --width 480
cd cpp_runtime && cmake -B build && cmake --build build --config Release
./build/nc_decode --model ../decoder.onnx --input clip.nec --benchmark
```

Requires CMake ≥ 3.18, a C++17 compiler and zlib. See [cpp_runtime/README.md](cpp_runtime/README.md).
The C++ runtime has not been compiled in CI yet.

## Development

```bash
pip install -e ".[dev,onnx]"
ruff check neurocodec eval demo tests
pytest
```

## Citation

```bibtex
@misc{narayana2026neurocodec,
  title  = {NeuroCodec: Neuromorphic-Inspired Spike-Driven Video Compression
            with Learned ConvLSTM Reconstruction},
  author = {Narayana, Chandini},
  year   = {2026},
  url    = {https://github.com/chandini-narayana/neurocodec},
  note   = {Work in progress}
}
```

---

## Links

Planned locations; not all of these are live yet (the package is not on PyPI and there is no arXiv id or released model).

-  **Live Demo:** [huggingface.co/spaces/chandininarayana/neurocodec](https://huggingface.co/spaces/chandininarayana/neurocodec)
-  **PyPI:** [pypi.org/project/neurocodec](https://pypi.org/project/neurocodec/)
-  **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
-  **Model:** [huggingface.co/chandininarayana/neurocodec](https://huggingface.co/chandininarayana/neurocodec)

---

**Copyright (c) 2026 Chandini Narayana — MIT License**
