# Publishing NeuroCodec

## PyPI

```bash
pip install -e ".[release]"
python -m build   # or: hatch build
twine check dist/*
twine upload --repository testpypi dist/*   # test first
twine upload dist/*                          # publish
```

## HuggingFace Spaces

1. Create Space at huggingface.co/new-space (SDK: Gradio)
2. Upload `demo/app.py` as `app.py`
3. Upload `demo/requirements.txt` as `requirements.txt`
4. Add example clips to `examples/` folder in the Space

## arXiv preprint

1. Compile `paper/neurocodec.tex` on Overleaf
2. Export PDF
3. Submit to arXiv cs.CV at arxiv.org/submit
4. Update `pyproject.toml` [project.urls] Paper field with arXiv URL

## Before publishing

- Train and add real decoder weights; run `eval/benchmark_rd.py` and replace the placeholder
  numbers in `paper/neurocodec.tex` and the README with measured ones.
- Replace the `arXiv` placeholders only once a real arXiv id exists.
- The demo Space needs `decoder_v1.pt` (otherwise it only shows the event visualisation).
