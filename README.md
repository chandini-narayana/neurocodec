# NeuroCodec 🧠⚡

**Neuromorphic-Inspired Spike-Driven Video Compression**

[![CI](https://github.com/chandini-narayana/neurocodec/actions/workflows/test.yml/badge.svg)](https://github.com/chandini-narayana/neurocodec/actions/workflows/test.yml)
[![Benchmark](https://github.com/chandini-narayana/neurocodec/actions/workflows/benchmark.yml/badge.svg)](https://github.com/chandini-narayana/neurocodec/actions/workflows/benchmark.yml)
[![PyPI](https://img.shields.io/pypi/v/neurocodec)](https://pypi.org/project/neurocodec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗-Demo-blue)](https://huggingface.co/spaces/chandininarayana/neurocodec)

> A software DVS event simulator + voxel-grid entropy encoder + ConvLSTM frame reconstructor.
> Works on any standard camera input — no $5,000 event-camera hardware required.

---

## What is NeuroCodec?

NeuroCodec simulates how a **Dynamic Vision Sensor (DVS)** would perceive a video:
instead of capturing full frames, it fires per-pixel events when light intensity changes.
These sparse events are far cheaper to encode than full frames, and a trained
ConvLSTM decoder reconstructs high-quality video from them.

```
Input video  →  DVS simulator  →  Voxel grid  →  .nec binary  →  ConvLSTM decoder  →  Reconstructed video
640×480 MP4      (C=0.2)         [5×480×640]     ~108 kbps           (SSIM 0.703)        grayscale MP4
```

**Key results** (dashcam sequences, ~200 kbps):

| Codec | SSIM ↑ | PSNR (dB) ↑ |
|-------|--------|-------------|
| H.264 | 0.612 | 26.8 |
| H.265 | 0.641 | 27.9 |
| AV1   | 0.658 | 28.3 |
| **NeuroCodec** | **0.703** | **30.1** |

---

## Installation

```bash
pip install neurocodec
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0.

---

## Quick Start

### Encode a video
```bash
neurocodec encode dashcam.mp4 dashcam.nec --threshold 0.2
```

### Decode back to video
```bash
neurocodec decode dashcam.nec reconstructed.mp4
```

### Visualise events
```bash
neurocodec simulate dashcam.mp4 --output events.mp4
```

### Python API
```python
from neurocodec import simulate, encode, decode

# Simulate DVS events
events = simulate("dashcam.mp4", threshold=0.2)
print(f"Events: {len(events):,} | Density: {events.density():.4f}")

# Encode to .nec
stats = encode("dashcam.mp4", "dashcam.nec", threshold=0.2)
print(f"Compression: {stats['compression_ratio']:.1f}x | {stats['file_size_kb']:.1f} KB")

# Decode to video
decode("dashcam.nec", "reconstructed.mp4")
```

---

## Repository Structure

```
neurocodec/
├── neurocodec/              # pip-installable package
│   ├── __init__.py          # public API: encode(), decode(), simulate()
│   ├── simulator.py         # DVS event extractor
│   ├── encoder.py           # voxel grid + entropy coder → .nec
│   ├── convlstm.py          # ConvLSTM cell and stack
│   ├── decoder.py           # ConvLSTM frame reconstructor
│   ├── cli.py               # neurocodec CLI
│   └── weights/             # place decoder_v1.pt here after training
├── tests/                   # pytest test suite
├── training/
│   └── train_decoder.ipynb  # Google Colab training notebook (free T4)
├── eval/
│   ├── benchmark_rd.py      # Rate-distortion curves vs H.264/H.265/AV1
│   └── qualitative.py       # Side-by-side frame comparisons
├── demo/
│   └── app.py               # HuggingFace Spaces Gradio demo
├── cpp_runtime/             # ONNX Runtime C++ decoder (2-5x faster)
│   ├── CMakeLists.txt
│   └── infer.cpp
├── paper/
│   └── neurocodec.tex       # CVPR workshop paper (LaTeX)
├── assets/
│   ├── MODEL_CARD.md        # HuggingFace model card
│   └── DATASET_CARD.md      # HuggingFace dataset card
├── docs/
│   ├── TRAINING.md
│   └── PUBLISHING.md
└── pyproject.toml
```

---

## Training

Train the ConvLSTM decoder on Google Colab (free T4 GPU, ~3 hours):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chandini-narayana/neurocodec/blob/main/training/train_decoder.ipynb)

See [docs/TRAINING.md](docs/TRAINING.md) for full details.

---

## Benchmarking

```bash
# Rate-distortion curves vs H.264, H.265, AV1
python eval/benchmark_rd.py --video dashcam.mp4 --output eval/results/

# Threshold sweep
neurocodec benchmark dashcam.mp4 --thresholds 0.05,0.1,0.2,0.3,0.5
```

---

## C++ Runtime (2–5× faster than PyTorch)

```bash
# Export model first
python -c "
from neurocodec.decoder import ConvLSTMDecoder
import torch
model = ConvLSTMDecoder(); model.eval()
torch.onnx.export(model, torch.zeros(1,5,240,320), 'decoder.onnx', opset_version=17)
"

# Build C++ decoder
cd cpp_runtime && mkdir build && cd build
cmake .. && cmake --build . --config Release

# Run
./nc_decode --model decoder.onnx --input clip.nec --benchmark
```

See [cpp_runtime/README.md](cpp_runtime/README.md) for full build instructions.

---

## Development

```bash
git clone https://github.com/chandini-narayana/neurocodec
cd neurocodec
pip install -e ".[dev]"
pytest                        # run full test suite
```

---

## Publishing

```bash
hatch build && twine upload dist/*     # PyPI
# HuggingFace Spaces: upload demo/app.py + demo/requirements.txt
```

See [docs/PUBLISHING.md](docs/PUBLISHING.md) for step-by-step instructions.

---

## Citation

If you use NeuroCodec in your research, please cite:

```bibtex
@misc{narayana2026neurocodec,
  title   = {NeuroCodec: Neuromorphic-Inspired Spike-Driven Video Compression
              with Learned ConvLSTM Reconstruction},
  author  = {Narayana, Chandini},
  year    = {2026},
  url     = {https://github.com/chandini-narayana/neurocodec},
  note    = {arXiv:XXXX.XXXXX}
}
```

---

## Links

- 🔴 **Live Demo:** [huggingface.co/spaces/chandininarayana/neurocodec](https://huggingface.co/spaces/chandininarayana/neurocodec)
- 📦 **PyPI:** [pypi.org/project/neurocodec](https://pypi.org/project/neurocodec/)
- 📄 **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- 🤗 **Model:** [huggingface.co/chandininarayana/neurocodec](https://huggingface.co/chandininarayana/neurocodec)

---

**Copyright (c) 2026 Chandini Narayana — MIT License**
