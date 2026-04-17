# Rage-Bait Detector

Python pipeline for detecting rage-bait social media posts with:

- dataset ingestion and annotation scaffolding
- an interactive mixed-file importer for Kaggle CSV, Parquet, and SQL schema files
- rule-based preprocessing and augmentation
- a vLLM-based weak-labeling pipeline with guided JSON decoding
- scikit-learn baselines for benchmark comparisons
- a PyTorch + BERT classifier with early stopping
- evaluation artifacts including classification reports and confusion matrices
- inference safeguards for empty, non-English, and overly long posts

## What Counts As Rage-Bait

Use the policy in [docs/annotation_guidelines.md](docs/annotation_guidelines.md) to keep labels consistent. The project assumes a binary label schema:

- `0`: not rage-bait
- `1`: rage-bait

The annotation workflow is designed for a 20,000+ post corpus even though this repo does not ship social platform data.

## Project Layout

```text
ragebait_detector/
  data/              # ingestion, cleaning, augmentation, dataset splits
  labeling/          # vLLM weak-labeling helpers
  models/            # baseline models and BERT classifier
  training/          # trainer, checkpoints, early stopping
  evaluation.py      # reports and confusion matrices
  inference.py       # runtime prediction edge-case handling
  pipeline.py        # CLI entrypoint
configs/
  default.yaml
docs/
  annotation_guidelines.md
scripts/
  interactive_import.py
  label_with_vllm.py
  analyze_labeled_csv.py
  balance_labeled_csv.py
  compress_labeled_csv.py
  run_pipeline.py
tests/
  test_labeled_csv_utils.py
  test_preprocessing.py
  test_unifier.py
  test_vllm_labeler.py
```

## Where To Put Files

Put your untouched Kaggle exports in `data/raw`.

- `data/raw`: original CSV, Parquet, and SQL/TXT files
- `data/unlabeled`: the compiled clean CSV that is ready for labeling
- `data/labeled`: vLLM labeling outputs
- `data/interim`: manifests, templates, and temporary artifacts

If the `.txt` file is only PostgreSQL schema SQL, that is still fine. The importer will inspect it, show the tables it found, and skip it if the file contains no actual row data.

## Dataset Acquisition Strategy

1. Export raw social posts from approved sources such as the X API, a licensed archive, or an internal moderation feed.
2. Normalize mixed CSV or JSONL exports into a common schema with `prepare-exports`.
3. Generate an annotation sheet with `build-annotation-template`.
4. Apply the annotation policy with at least two raters for ambiguous samples.
5. Merge labels back into the unified dataset with `merge-annotations`.
6. Run preprocessing, weak labeling, and training.

Expected unified columns:

- `post_id`
- `author_id`
- `created_at`
- `language`
- `text`
- `source`
- `label`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## CLI Usage

Interactively compile raw Kaggle files into one clean unlabeled CSV:

```bash
python3 scripts/interactive_import.py --config configs/default.yaml
```

The same flow is also available through the main CLI:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml interactive-import
```

Generate a mock labeled dataset for smoke testing:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml generate-mock-data --rows 500
```

Normalize raw exports:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml prepare-exports \
  --inputs data/raw/export_a.csv data/raw/export_b.jsonl
```
Create an annotation sheet:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml merge-annotations \
  --annotations data/interim/annotation_labels.csv
```

Merge completed annotations:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml merge-annotations \
  --annotations data/interim/annotation_labels.csv
```

Run preprocessing only:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml preprocess
```

Run the full training pipeline:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml run
```

Run baselines only:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml run --baselines-only
```

Label the compiled unlabeled CSV with vLLM. The default config uses `Qwen/Qwen2.5-3B-Instruct-AWQ` with AWQ quantization, sends one ChatML prompt per tweet, and passes the full flat prompt list to `llm.generate(...)` so vLLM can do continuous batching internally. By default it labels up to `50000` rows, uses seeded random sampling, and balances that sampling across the CSV `source` column so larger datasets do not dominate the batch:

```bash
python3 scripts/label_with_vllm.py --config configs/default.yaml --limit 500 --random-seed 42
```

Or through the main CLI:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml label-with-vllm --limit 500 --random-seed 42
```

## Large Labeled Dataset Utilities

The full weak-labeled export is stored as `data/labeled/vllm_all_qwen_ragebait_labels.csv.gz`. The raw CSV was about `157 MB`, while the gzip version is about `33 MB`, which makes it much easier to keep in the repo and push to GitHub.

The helper scripts below all assume the canonical raw path is `data/labeled/vllm_all_qwen_ragebait_labels.csv`. If that CSV is missing but the `.csv.gz` file exists, they automatically restore the raw CSV before continuing.

Compress the large labeled CSV:

```bash
python3 scripts/compress_labeled_csv.py \
  --input data/labeled/vllm_all_qwen_ragebait_labels.csv
```

This writes `data/labeled/vllm_all_qwen_ragebait_labels.csv.gz` and, by default, removes the uncompressed CSV. Pass `--keep-original` if you want both copies to remain on disk.

Analyze the labeled CSV at several confidence thresholds:

```bash
python3 scripts/analyze_labeled_csv.py \
  --input data/labeled/vllm_all_qwen_ragebait_labels.csv
```

This script reports:

- total row count
- valid labeled row count
- detected rage-bait row count
- overall rage-bait ratio
- rage-bait ratio and coverage at configurable confidence cutoffs

Create a balanced training CSV with a configurable class split:

```bash
python3 scripts/balance_labeled_csv.py \
  --input data/labeled/vllm_all_qwen_ragebait_labels.csv \
  --limit 32000 \
  --confidence-threshold 0.95 \
  --label-ratio 60/40
```

The balancer uses the `source`, `author_id`, `label`, and `confidence` columns together:

- rows below the confidence threshold are dropped
- rows are sampled to match the requested `label0/label1` ratio such as `50/50` or `60/40`
- within each label bucket, sampling is rotated across sources and authors so one large source or one prolific account does not dominate the output
- the output filename is generated from the arguments, for example `balanced_32000_c95_r60_40.csv`

## Findings From The 500k Weak Labels

On the current `507,682` row labeled file:

- total detected rage-bait rows: `31,072`
- overall rage-bait ratio: `6.12%`
- at `0.90` confidence: `466,165` rows retained and `29,369` rage-bait rows retained
- at `0.95` confidence: `102,104` rows retained and `12,946` rage-bait rows retained

The confidence values are fairly discrete in this dataset, so `0.92`, `0.93`, `0.94`, and `0.95` all currently produce the same retained pool. Using `0.95` is still the clearest choice because it documents the intent to keep only the highest-confidence group.

## Suggested BERT Training Set

For BERT training, the recommended starting point is:

- `data/labeled/balanced_32000_c95_r60_40.csv`

Why this is the default recommendation:

- `60/40` keeps more total rows than `50/50` without dropping the confidence threshold
- `0.95` keeps the high-confidence pool while still leaving enough positive examples for a useful supervised training set
- the output still covers all `8` sources seen in the high-confidence pool
- the balancing script also spreads samples across authors, which should reduce overfitting to a few prolific accounts

If you want a more class-symmetric alternative for comparison, use:

- `data/labeled/balanced_25000_c95_r50_50.csv`

## 32k BERT Workflow

This repo now includes a dataset-specific config for the recommended training file:

- `configs/bert_32k.yaml`

That config is wired to:

- read `data/labeled/balanced_32000_c95_r60_40.csv`
- write the cleaned dataset to `data/processed/processed_32000_c95_r60_40.csv`
- write run artifacts under `outputs/bert_32k/<timestamp>/`

The default model in that config is `bert-base-uncased`, but the training settings are intentionally conservative for local CPU-only runs:

```yaml
model:
  model_name: bert-base-uncased

training:
  batch_size: 8
  epochs: 1
```

The local config also disables minority-class augmentation because the balanced CSV is already intentionally class-shaped and ready for supervised training. If you have a real GPU and want a longer run, increase `training.epochs`, raise `model.max_length`, and adjust `training.batch_size` upward.

### Tuned BERT Config

There is also a tuned follow-up config:

- `configs/bert_32k_tuned.yaml`

This version keeps `bert-base-uncased` but changes the training behavior in one important way: it disables both the weighted sampler and the positive-class loss weight.

Why that matters:

- the 32k balanced file is only moderately imbalanced after English filtering
- the original trainer used both `WeightedRandomSampler` and `BCEWithLogitsLoss(pos_weight=...)`
- in practice that double-corrected the training signal toward class `1`
- the first BERT run collapsed into predicting ragebait far too often

The tuned config uses:

```yaml
training:
  batch_size: 8
  epochs: 3
  use_weighted_sampler: false
  use_pos_weight: false
```

Run it with:

```bash
./.venv/bin/python scripts/run_pipeline.py --config configs/bert_32k_tuned.yaml run --skip-baselines
```

On the current tuned run at `outputs/bert_32k_tuned/20260417_194800/`, BERT reached:

- accuracy: `0.8767`
- class `0` F1: `0.8937`
- class `1` F1: `0.8533`

That is slightly better overall than the best classic baseline observed on the earlier 32k runs, where the clean-text SVC reached about `0.8718` accuracy.

### 1. Preprocess The 32k Dataset

```bash
./.venv/bin/python scripts/run_pipeline.py --config configs/bert_32k.yaml preprocess
```

This writes:

- `data/processed/processed_32000_c95_r60_40.csv`
- `outputs/bert_32k/preprocess_summary.json`

The preprocessing step:

- normalizes labels to `0` or `1`
- cleans text by lowercasing and replacing URLs, mentions, emojis, and numbers
- detects language from the raw text and drops unsupported rows when `drop_non_english: true`
- removes empty or effectively content-free posts

### 2. Run Baselines

```bash
./.venv/bin/python scripts/run_pipeline.py --config configs/bert_32k.yaml run --baselines-only
```

This creates a timestamped run directory such as:

- `outputs/bert_32k/20260417_191055/`

Inside the `baselines/` folder you get JSON metrics plus confusion matrices for:

- logistic regression
- SVC
- Gaussian Naive Bayes
- decision tree

Each baseline is trained on both raw text and cleaned text so you can compare whether preprocessing helps classic sparse-text models.

### 3. Train The BERT Classifier

```bash
./.venv/bin/python scripts/run_pipeline.py --config configs/bert_32k.yaml run --skip-baselines
```

This writes BERT artifacts under the timestamped run directory:

- `bert/checkpoint.pt`
- `bert/test_metrics.json`
- `bert/training_history.json`
- `bert/test_confusion_matrix.png`
- `bert/artifacts.json`

The trainer:

- stratifies train, validation, and test splits by label
- tokenizes cleaned text
- uses weighted sampling plus `BCEWithLogitsLoss`
- keeps the best checkpoint by validation F1 for class `1`

On the current local `bert_32k.yaml` run, the classic baselines still outperform the one-epoch BERT model. Treat that config as a reproducible starting point. For the stronger follow-up run, use `configs/bert_32k_tuned.yaml`.

### 4. Test A Trained Checkpoint

Use the helper script:

- `scripts/test_trained_bert.py`

Score a single post with the latest checkpoint under `outputs/bert_32k/`:

```bash
./.venv/bin/python scripts/test_trained_bert.py \
  --config configs/bert_32k.yaml \
  --text "this post is clearly trying to provoke angry replies"
```

Score several posts explicitly against a known run:

```bash
./.venv/bin/python scripts/test_trained_bert.py \
  --config configs/bert_32k.yaml \
  --run-dir outputs/bert_32k/<timestamp> \
  --text "neutral update about the match tonight" \
  --text "everyone who disagrees is pathetic"
```

Score one post per line from a file:

```bash
./.venv/bin/python scripts/test_trained_bert.py \
  --config configs/bert_32k.yaml \
  --text-file data/interim/sample_posts.txt
```

By default, `scripts/test_trained_bert.py` now assumes you want to score the text even if language detection is noisy. In other words, it bypasses the inference-time language gate unless you explicitly turn that check back on.

```bash
./.venv/bin/python scripts/test_trained_bert.py \
  --config configs/bert_32k_tuned.yaml \
  --run-dir outputs/bert_32k_tuned/<timestamp> \
  --text "your English text here"
```

If you want the original strict language gate back for a test run, use:

```bash
./.venv/bin/python scripts/test_trained_bert.py \
  --config configs/bert_32k_tuned.yaml \
  --run-dir outputs/bert_32k_tuned/<timestamp> \
  --no-force-english \
  --text "your text here"
```

This only affects the test script. It does not retrain the model or change the saved checkpoint. It simply controls whether inference-time `drop_non_english` rejection is bypassed.

The script prints JSON with:

- the checkpoint path used
- whether forced-English scoring was enabled
- the predicted label
- confidence
- class probabilities
- any rejection reason such as empty content or unsupported language
- the number of chunks scored for long posts

## Architecture Decisions

The system separates data normalization, annotation merge, preprocessing, weak labeling, modeling, training, and inference so the pipeline can be scheduled and audited stage by stage. That makes it easier to swap data sources, re-run only the preprocessing step after guideline updates, and compare classic baselines against the BERT model on the exact same split.

The interactive importer intentionally keeps a person in the loop because Kaggle exports rarely agree on column names or completeness. Instead of hardcoding a single schema, the importer lists the files it finds, previews columns and sample rows, lets you select row ranges, asks for a display name for each source, and writes a canonical unlabeled CSV with `post_id, author_id, created_at, language, text, source`.

The vLLM labeler is a separate offline batch stage optimized for CUDA GPUs. It completely bypasses HTTP and manual chunking, formats every tweet as its own Qwen ChatML prompt, and relies on vLLM guided decoding with `guided_json` so the model is constrained to emit JSON objects matching the labeling schema.

The BERT classifier uses a pre-trained encoder plus a small ANN head with dropout. Training uses `BCEWithLogitsLoss` for numerical stability with class imbalance handling, while inference exposes softmax-style two-class probabilities derived from the model logit. This keeps the binary objective simple without losing calibrated class probabilities for downstream consumers.

Baseline models are trained on both raw text and cleaned text so you can quantify whether preprocessing helps or hurts simpler models. Logistic regression and linear-kernel SVC provide strong sparse-text benchmarks, while Gaussian Naive Bayes and decision trees act as weaker but interpretable references.

## Edge-Case Handling

- Empty or media-only posts are mapped to `[empty_post]` and excluded from training.
- Unsupported languages are detected and can be dropped when the model is English-only.
- URLs, mentions, and emojis are replaced with identifier tokens rather than deleted blindly.
- Very long posts are chunked at inference time using tokenizer windows and aggregated into one prediction.

## Scaling To Real-Time Feeds

For real-time social streams, run preprocessing and inference as separate stateless services behind a queue such as Kafka, Kinesis, or Pub/Sub. A lightweight ingestion worker can normalize raw platform payloads and push them onto a feature queue; batched GPU inference workers can then score posts in micro-batches for higher throughput.

Store predictions, model version, feature hashes, and confidence scores in an append-only analytics table so you can audit false positives, measure drift, and trigger relabeling jobs. In production, you would also add model registry support, automated retraining schedules, feature and label drift monitors, and a fallback policy for non-English or low-confidence content.

## Local Verification

The repo includes unit tests for preprocessing, mixed-file import helpers, and vLLM labeling logic. Run the focused suite with:

```bash
python3 -m pytest tests/test_vllm_labeler.py tests/test_preprocessing.py tests/test_unifier.py
```
