# Publishing NeuroCodec

## PyPI

```bash
pip install hatch twine
hatch build
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
