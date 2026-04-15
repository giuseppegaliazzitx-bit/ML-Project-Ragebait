# Rage-Bait Detector

Python pipeline for detecting rage-bait social media posts with:

- dataset ingestion and annotation scaffolding
- an interactive mixed-file importer for Kaggle CSV, Parquet, and SQL schema files
- rule-based preprocessing and augmentation
- an Ollama-based weak-labeling pipeline using tool calls
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
  labeling/          # Ollama weak-labeling helpers
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
  label_with_ollama.py
  run_pipeline.py
tests/
  test_preprocessing.py
  test_unifier.py
  test_ollama_labeler.py
```

## Where To Put Files

Put your untouched Kaggle exports in [data/raw](/Users/red/dev/ML-Project-Ragebait/data/raw).

- `data/raw`: original CSV, Parquet, and SQL/TXT files
- `data/unlabeled`: the compiled clean CSV that is ready for labeling
- `data/labeled`: Ollama labeling outputs
- `data/interim`: manifests, templates, and temporary artifacts

If the `.txt` file is only PostgreSQL schema SQL, that is still fine. The importer will inspect it, show the tables it found, and skip it if the file contains no actual row data.

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

Label the compiled unlabeled CSV with Ollama. The default config uses `qwen2.5:3b-instruct-q4_K_M`, batches requests in groups of `10`, and shows a progress bar while it runs. You can override the model, worker count, and batch size on the command line:

```bash
python3 scripts/label_with_ollama.py --config configs/default.yaml --workers 1 --batch-size 10
```

Or through the main CLI:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml label-with-ollama
```

## Architecture Decisions

The system separates data normalization, annotation merge, preprocessing, modeling, training, and inference so the pipeline can be scheduled and audited stage by stage. That makes it easier to swap data sources, re-run only the preprocessing step after guideline updates, and compare classic baselines against the BERT model on the exact same split.

The interactive importer intentionally keeps a person in the loop because Kaggle exports rarely agree on column names or completeness. Instead of hardcoding a single schema, the importer lists the files it finds, previews columns and sample rows, lets you select row ranges, asks for a display name for each source, and writes a canonical unlabeled CSV with `post_id, author_id, created_at, language, text, source`.

The Ollama labeler is a separate stage so you can inspect the compiled unlabeled data before generating weak labels. It deduplicates repeated text, sends micro-batched requests to the local Ollama chat API, asks the configured chat model for native JSON output instead of tool calls, and stores both the boolean decision and a numeric `0/1` label to make downstream review and training simpler.

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

The repo includes unit tests for preprocessing, mixed-file import helpers, and Ollama JSON parsing. Full model training and Parquet import require installing the dependencies declared in [pyproject.toml](/Users/red/dev/ML-Project-Ragebait/pyproject.toml).
