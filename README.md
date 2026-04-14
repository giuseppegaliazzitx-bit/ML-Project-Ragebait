# Rage-Bait Detector

Python pipeline for detecting rage-bait social media posts with:

- dataset ingestion and annotation scaffolding
- rule-based preprocessing and augmentation
- scikit-learn baselines for benchmark comparisons
- a PyTorch + BERT classifier with early stopping
- evaluation artifacts including classification reports and confusion matrices
- inference safeguards for empty, non-English, and overly long posts

## What Counts As Rage-Bait

Use the policy in [docs/annotation_guidelines.md](/Users/red/dev/ML-Project-Ragebait/docs/annotation_guidelines.md) to keep labels consistent. The project assumes a binary label schema:

- `0`: not rage-bait
- `1`: rage-bait

The annotation workflow is designed for a 20,000+ post corpus even though this repo does not ship social platform data.

## Project Layout

```text
ragebait_detector/
  data/              # ingestion, cleaning, augmentation, dataset splits
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
  run_pipeline.py
tests/
  test_preprocessing.py
```

## Dataset Acquisition Strategy

1. Export raw social posts from approved sources such as the X API, a licensed archive, or an internal moderation feed.
2. Normalize mixed CSV or JSONL exports into a common schema with `prepare-exports`.
3. Generate an annotation sheet with `build-annotation-template`.
4. Apply the annotation policy with at least two raters for ambiguous samples.
5. Merge labels back into the unified dataset with `merge-annotations`.
6. Run preprocessing and training.

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
python3 scripts/run_pipeline.py --config configs/default.yaml build-annotation-template
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

## Architecture Decisions

The system separates data normalization, annotation merge, preprocessing, modeling, training, and inference so the pipeline can be scheduled and audited stage by stage. That makes it easier to swap data sources, re-run only the preprocessing step after guideline updates, and compare classic baselines against the BERT model on the exact same split.

The BERT classifier uses a pre-trained encoder plus a small ANN head with dropout. Training uses `BCEWithLogitsLoss` for numerical stability with class imbalance handling, while inference exposes softmax-style two-class probabilities derived from the model logit. This keeps the binary objective simple without losing calibrated class probabilities for downstream consumers.

Baseline models are trained on both raw text and cleaned text so you can quantify whether preprocessing helps or hurts simpler models. Logistic regression and linear-kernel SVC provide strong sparse-text benchmarks, while Gaussian Naive Bayes and decision trees act as weaker but interpretable references.

## Edge-Case Handling

- Empty or media-only posts are mapped to `[empty_post]` and excluded from training.
- Unsupported languages are detected and can be dropped when the model is English-only.
- URLs, mentions, hashtags, and emojis are replaced with identifier tokens rather than deleted blindly.
- Very long posts are chunked at inference time using tokenizer windows and aggregated into one prediction.

## Scaling To Real-Time Feeds

For real-time social streams, run preprocessing and inference as separate stateless services behind a queue such as Kafka, Kinesis, or Pub/Sub. A lightweight ingestion worker can normalize raw platform payloads and push them onto a feature queue; batched GPU inference workers can then score posts in micro-batches for higher throughput.

Store predictions, model version, feature hashes, and confidence scores in an append-only analytics table so you can audit false positives, measure drift, and trigger relabeling jobs. In production, you would also add model registry support, automated retraining schedules, feature and label drift monitors, and a fallback policy for non-English or low-confidence content.

## Local Verification

The repo includes stdlib unit tests for preprocessing. Full model training requires installing the ML dependencies declared in [pyproject.toml](/Users/red/dev/ML-Project-Ragebait/pyproject.toml).
