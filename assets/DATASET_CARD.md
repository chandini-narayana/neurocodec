---
license: cc-by-4.0
task_categories:
  - video-classification
tags:
  - event-camera
  - dvs
  - neuromorphic
  - video-compression
---

# NeuroCodec Evaluation Dataset

Derived evaluation set for rate-distortion benchmarking of NeuroCodec.

## Content

- 8 dashcam video sequences (30 fps, 640×480)
- 8 DAVIS240C sequences (30 fps, 240×180)
- Pre-encoded .nec files at 6 threshold values per sequence
- Corresponding H.264/H.265/AV1 encodings for baseline comparison

## License

Original DAVIS240C sequences: CC-BY 4.0 (rpg.ifi.uzh.ch)  
Dashcam footage and .nec encodings: MIT (Chandini Narayana, 2026)

## Usage

```python
from datasets import load_dataset
ds = load_dataset("chandininarayana/neurocodec-eval")
```
