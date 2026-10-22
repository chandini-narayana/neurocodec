# Training the ConvLSTM Decoder

## Overview

The decoder is trained on Google Colab (free T4 GPU).
Full training takes ~3 hours for 50 epochs on a few thousand clips.

Open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chandini-narayana/neurocodec/blob/main/training/train_decoder.ipynb)

## Dataset

The decoder trains on ordinary videos; events are simulated from them. The notebook
downloads **UCF101** (~6.9 GB, 13k clips at 320x240, research use,
https://www.crcv.ucf.edu/data/UCF101.php) and keeps a random subset of `MAX_UCF_CLIPS`
(default 3000). Add your own clips (best: footage like your target, e.g. dashcam) to
`/content/videos/extra`. Other sources: Xiph.org test sequences, Pexels/Pixabay free
videos, BDD100K (free account). Hold a few clips out entirely for the final benchmark.

## Quick start (local, if you have a GPU)

```bash
pip install -e ".[train]"
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

## Notes on the training setup

- Each training item is `seq_len` consecutive frames; the voxel for frame *i* is paired with
  frame *i* (voxels use a sliding window of the last `n_bins` frames, exactly as the encoder
  does). The ConvLSTM state is reset per sequence and gradients flow through all `seq_len` steps.
- 10 % of sequences are held out; the notebook saves the checkpoint with the best *validation* SSIM.
- Without video files the notebook falls back to a synthetic gradient — that only smoke-tests
  the code and will not give a useful model.
- Events carry no absolute brightness; expect poor results on static scenes unless key-frames
  are added to the codec.
