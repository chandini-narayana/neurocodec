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

**Author:** Chandini Narayana (2026) · **License:** MIT

> **Status: weights not released yet.** This card describes the intended model. No
> trained checkpoint or measured quality numbers exist yet; the table that used to be
> here contained placeholders and was removed.

## What is this?

The ConvLSTM frame reconstructor for
[NeuroCodec](https://github.com/chandini-narayana/neurocodec). Given the sparse voxel
grid `[B×H×W]` of simulated DVS events (decoded from a `.nec` file) it reconstructs the
corresponding grayscale frame. The ConvLSTM state carries over between frames.

## Usage (once weights exist)

```python
from neurocodec import decode
decode("clip.nec", "reconstructed.mp4", weights_path="decoder_v1.pt")
```

```python
import torch
from neurocodec.decoder import load_decoder

model = load_decoder("decoder_v1.pt", n_bins=5)   # raises if the file is missing
frame = model(torch.zeros(1, 5, 240, 320))        # (1, 1, 240, 320)
model.reset_state()                               # between unrelated clips
```

## Training (planned, see `training/train_decoder.ipynb`)

- **Data:** any videos, or DAVIS240C clips; events are simulated with `DVSSimulator`
- **Loss:** 0.5×L1 + 0.5×VGG16 perceptual (relu2_2, ImageNet-normalised)
- **Optimizer:** AdamW, lr=1e-4, cosine annealing; 10 % of sequences held out for validation SSIM
- **Hardware:** Google Colab T4

## Limitations

- Events encode changes only; static regions and absolute brightness are not recoverable
  from events alone.
- Grayscale output only.
- Compression of the `.nec` container is not yet competitive with H.264 (see the README).

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
