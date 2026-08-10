# Training the ConvLSTM Decoder

## Overview

The decoder is trained on Google Colab (free T4 GPU).
Full training takes ~3 hours for 50 epochs on the DAVIS240C dataset.

Open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chandini-narayana/neurocodec/blob/main/training/train_decoder.ipynb)

## Dataset

**DAVIS240C** — Dynamic and Active-pixel Vision Sensor dataset.
- 240×180 resolution, 40+ sequences, ~6.5 GB
- Download: https://rpg.ifi.uzh.ch/davis_data.html
- License: Creative Commons Attribution 4.0

## Quick start (local, if you have a GPU)

```bash
pip install neurocodec[dev]
jupyter notebook training/train_decoder.ipynb
```

## Saving weights

The notebook saves `decoder_v1.pt` to Google Drive.
Copy it to `neurocodec/weights/decoder_v1.pt` for bundling with the package.

To publish on HuggingFace Hub:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='neurocodec/weights/decoder_v1.pt',
    path_in_repo='decoder_v1.pt',
    repo_id='chandininarayana/neurocodec',
    repo_type='model',
)
"
```
