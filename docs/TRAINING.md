# Training the ConvLSTM Decoder

## Overview

The decoder is trained on Google Colab (free T4 GPU).
Training time depends on the GPU and on `steps_per_epoch` / `n_epochs` in the config cell; the loop
prints seconds per step so you can estimate it. With the defaults (20 epochs x 300 steps) expect
tens of minutes to a few hours on a T4. Without a GPU it is impractically slow.

Open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chandini-narayana/neurocodec/blob/main/training/train_decoder.ipynb)

## Dataset

The decoder trains on ordinary videos; events are simulated from them. The notebook's
`SOURCE` setting picks the data:

- `'subset'` (default): 405 UCF101 clips from 10 classes, 171 MB
  (Hugging Face `sayakpaul/ucf101-subset`). Enough for a first model and a quick test.
- `'full'`: all of UCF101, a 6.9 GB RAR (13k clips at 320x240, research use,
  https://www.crcv.ucf.edu/data/UCF101.php); `MAX_UCF_CLIPS` keeps a random subset.
- `'none'`: only your own clips.

Add your own clips (best: footage like your target, e.g. dashcam) to `/content/videos/extra`.
Other sources: Xiph.org test sequences, Pexels/Pixabay free videos, BDD100K (free account).
Hold a few clips out entirely for the final benchmark.

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
