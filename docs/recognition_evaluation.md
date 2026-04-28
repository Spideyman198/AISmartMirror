# Recognition system: evaluation and tuning order

This doc matches the **step-by-step** improvement plan for AISmartMirror face recognition (CNN + baseline embedding).

## Automation (holdout split, leak check, full pipeline)

- **Holdout split** (move a fraction from `data/cnn_faces/<user>/` to `data/eval_benchmark/<user>/` before `prepare`):
  - `python scripts/split_holdout_benchmark.py --ratio 0.25 --move`
  - Strangers folder → `eval_benchmark/_strangers/`: `--strangers-from path/to/crops`
- **Leakage check** (same file must not appear in benchmark and train/val/raw collection):
  - `python scripts/check_benchmark_leakage.py`
- **Full pipeline** (prepare with `--clean`, leak check, train, val metrics, benchmark):
  - `python scripts/run_cnn_eval_pipeline.py`

## 1) Benchmark layout (same images for both recognizers)

Create a **holdout** set not used in CNN training:

```
data/eval_benchmark/
  areej/
  kinan/
  ...
  _strangers/          # optional: people not in profiles.json / CNN classes
```

Run:

```bash
python scripts/evaluate_recognition_comparison.py --benchmark-dir data/eval_benchmark
```

Optional: `--json-out metrics/recognition_compare.json`

**Metrics to track**

| Metric | Meaning |
|--------|--------|
| **Correct ID** (known folders) | Predicted user matches folder name |
| **Wrong ID** | Predicted another enrolled user |
| **Unknown** (known folders) | Reject / unknown (false negative for ID) |
| **Stranger → unknown** | Good (true negative) |
| **Stranger → named** | Bad (false accept) |

Compare **CNN** vs **Embedding** on the **same files**.

## 2) Live CNN verification (already in repo)

- `python scripts/debug_cnn_pipeline.py` — model path, mapping, top-3 on a val or live crop
- `python scripts/test_cnn_live.py --debug` — full preprocessing logs
- `scripts/test_cnn_live.py` — `CNN_CONFIDENCE_THRESHOLD`, `CNN_MIN_CLASS_MARGIN` from `.env`

## 3) Tuning order (do in this order)

1. **Verify pipeline**: `debug_cnn_pipeline.py` on val + one live crop; confirm top-3 sane.
2. **Crop quality**: `--min-blur 35–50`, `--min-det 0.45`, optional `--crop-margin 0.05` (if train used tight crops, margin is optional).
3. **Unknown handling**: `CNN_CONFIDENCE_THRESHOLD` ↑ (e.g. 0.62→0.68), `CNN_MIN_CLASS_MARGIN` ↑ (0.15→0.18), optional `CNN_MAX_SOFTMAX_ENTROPY` (~1.4).
4. **Stability**: `interval` 3–4, `confirm` 5–6 (already defaults in `test_cnn_live.py`); keep **left-right sort** on for multi-face.
5. **Re-collect data** (see below), then retrain CNN with **moderate** aug (`--aug full` is OK; avoid extreme extra aug without more real data).

## 4) Dataset collection (next round)

- **More sessions**: different days/times (lighting).
- **Pose**: up/down/left/right, not only frontal.
- **Distance**: near and far mirror distance.
- **Strangers / negatives**: folder `_strangers` for benchmark; for training an **unknown** class, add many non-user crops (optional 7th class).
- **Confused pairs**: extra balanced samples for pairs the model mixes.

## 5) Training augmentations

Use **`--aug full`** in `train_cnn_recognizer.py` (moderate–strong, already tuned). Avoid piling on **extra** offline aug **without** new real sessions—domain gap matters more than synthetic volume.

## 6) Alignment (optional)

**Face alignment** (eye-based warp) before CNN can help but adds complexity and must match train/infer. Defer until benchmark + crop QC are exhausted.
