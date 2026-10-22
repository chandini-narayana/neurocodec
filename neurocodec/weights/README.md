# Model Weights

Place the trained checkpoint here as `decoder_v1.pt` (git-ignored), or pass
`--weights path/to/decoder_v1.pt` to the CLI.

## Training

Run `training/train_decoder.ipynb` on Google Colab (free T4 GPU). It saves the best
validation-SSIM checkpoint to Google Drive as `decoder_v1.pt`; copy it here. The checkpoint
stores `model_state_dict` plus the training `config` (used to rebuild the right layer sizes).

## Without weights

`neurocodec decode` and `decode()` raise `FileNotFoundError` when no weights exist — a
random model only produces a constant gray image. To test the plumbing anyway use
`--allow-random` / `allow_random_weights=True`.

Weights are not published yet; there is no download URL.
