---
language: en
license: mit
tags:
  - video-compression
  - neuromorphic
  - event-camera
  - convlstm
  - dvs
library_name: neurocodec
pipeline_tag: video-to-video
---

# NeuroCodec — ConvLSTM Decoder v1

**Model:** `chandininarayana/neurocodec`  
**Author:** Chandini Narayana (2026)  
**License:** MIT

## What is this?

This is the ConvLSTM video frame reconstructor for
[NeuroCodec](https://github.com/chandini-narayana/neurocodec),
a neuromorphic-inspired video codec.

Given a sparse voxel grid of DVS events (shape `[B×H×W]`),
it reconstructs the corresponding grayscale video frame.

## Usage

```python
from neurocodec import decode
decode("clip.nec", "reconstructed.mp4")
```

Or with the model directly:

```python
import torch
from neurocodec.decoder import load_decoder

model = load_decoder()   # downloads weights automatically
voxel = torch.zeros(1, 5, 240, 320)   # batch=1, bins=5
frame = model(voxel)     # (1, 1, H, W)
```

## Training

- **Dataset:** DAVIS240C (32 sequences) + dashcam footage
- **Loss:** 0.5×L1 + 0.5×VGG perceptual (relu2_2)
- **Optimizer:** AdamW, lr=1e-4, cosine annealing, 50 epochs
- **Hardware:** Google Colab T4 GPU, ~3 hours

## Performance

| Codec | SSIM @200kbps | PSNR (dB) |
|-------|--------------|-----------|
| H.264 | 0.612 | 26.8 |
| H.265 | 0.641 | 27.9 |
| AV1   | 0.658 | 28.3 |
| **NeuroCodec** | **0.703** | **30.1** |

*On dashcam/drone sequences. See paper for full R-D curves.*

## Citation

```bibtex
@misc{narayana2026neurocodec,
  title  = {NeuroCodec: Neuromorphic-Inspired Spike-Driven Video Compression
             with Learned ConvLSTM Reconstruction},
  author = {Narayana, Chandini},
  year   = {2026},
  url    = {https://github.com/chandini-narayana/neurocodec},
}
```
