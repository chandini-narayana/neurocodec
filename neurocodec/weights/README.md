# Model Weights

Place your trained model checkpoint here as `decoder_v1.pt`.

## Training

Run `training/train_decoder.ipynb` on Google Colab (free T4 GPU) to train the
ConvLSTM decoder.  The notebook saves checkpoints to Google Drive and outputs
a final `decoder_v1.pt` that you copy here.

## Download pretrained weights

Once published, weights will be available on HuggingFace Hub:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('chandininarayana/neurocodec', 'decoder_v1.pt', local_dir='neurocodec/weights/')
"
```

## Without weights

The decoder still runs with random initialisation — useful for testing the
pipeline end-to-end.  Reconstructed frames will look noisy until training
is complete.
