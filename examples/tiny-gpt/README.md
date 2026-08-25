# tiny-gpt

A small decoder-only (GPT-style) character-level language model trained on
Tiny Shakespeare with ember. ~4.8M parameters: `Embedding ->
PositionalEncoding -> TransformerEncoder(causal=True) -> Linear` head,
optimized with AdamW, batched with `ember.data.DataLoader`.

## Files

- `train.py` — downloads the dataset (first run only), trains the model,
  writes `losses.csv` and `sample.txt`.
- `plot_loss.py` — plots `losses.csv` into `loss_curve.png`.
- `loss_curve.png` — train/val loss curve of the reference run.
- `sample.txt` — text sampled from the trained model (temperature 0.8,
  prompt `"\nKING:"`).

## Run

```bash
# CUDA build (12.4 toolchain); training defaults to the visible GPU
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=7   # pick a GPU

uv run python examples/tiny-gpt/train.py
uv run --with matplotlib python examples/tiny-gpt/plot_loss.py
```

Key flags: `--steps 2000 --batch-size 64 --block-size 256 --dim 256 --heads 8
--layers 6 --lr 1e-3 --weight-decay 0.1` (all shown with their defaults).

## Notes

- Reference run: val loss reaches **~1.28 (perplexity ~3.6)** around step
  1750-2000, starting from 3.56. The model learns Shakespeare's structure —
  speaker names, dialogue formatting, blank-verse-ish rhythm.
- Longer runs overfit: at 6000 steps train loss drops to ~0.29 but val loss
  climbs back to ~2.76 (no dropout in the encoder). If you want to train
  longer, that is the first thing to add.
