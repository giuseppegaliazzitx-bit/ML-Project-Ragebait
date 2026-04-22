# Iteration 1 Report

## Purpose

This document records the end-to-end experiment path we completed in this repo for the **binary rage-bait classifier**. That binary system is now the project's **Baseline Model**.

The next project iteration will be a new **Primary Model** trained from scratch as a **single-dataset, multi-class classifier** with these labels:

- `Normal`
- `Profanity`
- `Trolling`
- `Derogatory`
- `Hate Speech`

This roadmap exists so we can compare the future primary model against a precise, artifact-backed baseline instead of relying on memory.

## Scope and Ground Rules

- The baseline task was **binary classification**:
  - `0 = not rage-bait`
  - `1 = rage-bait`
- The baseline labels were **weak labels**, not gold human labels.
- The final reported baseline model is the **tuned BERT run** in `outputs/bert_32k_tuned/20260417_194800/`.
- Where the exact CLI invocation was not preserved in an artifact, this document marks the step as **reconstructed from saved files and configs**.

## 1. Problem Definition

The original project goal was to detect whether a short social media post was intentionally written to provoke anger, dogpiles, or hostile engagement. The labeling policy is documented in `docs/annotation_guidelines.md`.

Operational label schema:

- `1`: rage-bait
- `0`: not rage-bait

Important distinction from the start: the task was not "toxicity detection" in general. The positive class was meant to capture **intentional engagement baiting**, not every angry or offensive post.

## 2. Raw Data Collection and Unification

### 2.1 Raw source files used

The saved import manifest in `data/interim/source_manifest.json` shows that the project combined these raw files into a single unified dataset:

| Raw file | Source name in unified data | Rows imported |
| --- | --- | ---: |
| `data/raw/AllTweets.csv` | `AllTweets` | 88,625 |
| `data/raw/common_authors_data - general-anon.csv` | `common-annon` | 894 |
| `data/raw/common_authors_data - israel-hamas-anon.csv` | `isreal-hamas-anon` | 434 |
| `data/raw/common_authors_data - ukraine-anon.csv` | `ukraine-anon` | 202 |
| `data/raw/common_authors_data - vaccine-anon.csv` | `vaccine-anon` | 145 |
| `data/raw/sample_import.csv` | `sample_import` | 50 |
| `data/raw/train-00000-of-00001.parquet` | `twtemotion` | 416,809 |
| `data/raw/twitter_toxic_tweets.csv` | `twitter_toxic_tweets` | 31,962 |

The manifest also records `x_db.txt` as selected and then skipped with `unsupported_sql_payload`.

### 2.2 Unified dataset construction

The raw files were compiled through the interactive importer into a common schema:

- `post_id`
- `author_id`
- `created_at`
- `language`
- `text`
- `source`

Artifact-backed outputs:

- Unified CSV: `data/unlabeled/unified_unlabeled_posts.csv`
- Import manifest: `data/interim/source_manifest.json`

Recorded unification result:

- Rows written: `507,682`
- Duplicate rows removed: `31,439`

The interactive importer deduplicated rows by this signature:

- `author_id`
- `created_at`
- `language`
- `text`
- `source`

After deduplication, `post_id` values were renumbered.

## 3. Weak Labeling Stage

### 3.1 Weak-label model used

Weak labeling used the vLLM pipeline configured in `configs/default.yaml` and `ragebait_detector/labeling/vllm_labeler.py`.

Model configuration:

- Model: `Qwen/Qwen2.5-3B-Instruct-AWQ`
- Quantization: `awq`
- Temperature: `0.0`
- Max model length: `1024`
- Source-balanced random sampling: enabled in config
- Output format: guided JSON with
  - `is_ragebait`
  - `confidence`
  - `reason`

The system prompt instructed the model to decide whether each post was rage-bait and to return only JSON.

### 3.2 Weak-label output

Saved outputs:

- `data/labeled/vllm_all_qwen_ragebait_labels.csv`
- `data/labeled/vllm_all_qwen_ragebait_labels.csv.gz`

The compressed copy was kept because the raw CSV was much larger and harder to store in the repo.

### 3.3 What the weak labels looked like

`scripts/analyze_labeled_csv.py` on `data/labeled/vllm_all_qwen_ragebait_labels.csv` reported:

- Total rows: `507,682`
- Valid labeled rows: `507,682`
- Detected rage-bait rows: `31,072`
- Overall rage-bait ratio: `6.12%`

High-confidence retention points:

| Confidence threshold | Rows kept | Rage-bait rows | Rage-bait ratio |
| --- | ---: | ---: | ---: |
| `0.90` | 466,165 | 29,369 | 6.30% |
| `0.92` | 102,104 | 12,946 | 12.68% |
| `0.95` | 102,104 | 12,946 | 12.68% |

The `0.92` and `0.95` pools were identical in this dataset. That matters because the final training set was chosen from this high-confidence slice.

### 3.4 Important limitation of this stage

This was the single biggest limitation of the baseline project:

- the BERT model was trained on **LLM-produced weak labels**
- the reported evaluation measures **agreement with those weak labels**
- it does **not** establish human-grounded truth

## 4. Balanced Training-Set Selection

### 4.1 Why balancing was needed

The full weak-labeled file was highly imbalanced:

- only `6.12%` of rows were labeled rage-bait overall

Training directly on that file would have made the supervised stage too skewed toward the negative class. A smaller, higher-confidence, class-shaped training set was created instead.

### 4.2 How balancing worked

The balancing utility was `scripts/balance_labeled_csv.py`, backed by `ragebait_detector/utils/labeled_csv.py`.

Selection logic:

1. Keep only `labeling_status == ok`
2. Keep only rows with valid `label` and `confidence`
3. Drop rows below the chosen confidence threshold
4. Sample to the requested class ratio
5. Within each class, rotate sampling across `source` and `author_id` so one source or prolific account does not dominate

### 4.3 Balanced datasets produced

The repo contains these generated candidate training files:

- `data/labeled/balanced_25000_c92_r50_50.csv`
- `data/labeled/balanced_25000_c95_r50_50.csv`
- `data/labeled/balanced_30000_c92_r60_40.csv`
- `data/labeled/balanced_32000_c92_r60_40.csv`
- `data/labeled/balanced_32000_c95_r60_40.csv`

### 4.4 Final training file selected

The final baseline training file was:

- `data/labeled/balanced_32000_c95_r60_40.csv`

Why this one was chosen:

- it stayed inside the high-confidence pool
- it preserved more total rows than the `50/50` alternatives
- it kept a meaningful positive class without collapsing the dataset size

Observed composition of `balanced_32000_c95_r60_40.csv`:

- Total rows: `32,000`
- Label `0`: `19,200`
- Label `1`: `12,800`
- Confidence range: `0.95` to `1.0`

Source distribution:

| Source | Rows |
| --- | ---: |
| `twtemotion` | 13,305 |
| `AllTweets` | 9,106 |
| `twitter_toxic_tweets` | 8,816 |
| `common-annon` | 353 |
| `isreal-hamas-anon` | 204 |
| `ukraine-anon` | 98 |
| `vaccine-anon` | 80 |
| `sample_import` | 38 |

## 5. Binary Baseline Pipeline

### 5.1 Preprocessing rules

The binary training pipeline used `ragebait_detector/data/preprocessing.py`.

Preprocessing behavior:

- normalize labels to `0/1`
- detect language with `langdetect`
- lowercase text
- replace URLs with `[url]`
- replace user mentions with `[user]`
- replace emojis with `[emoji]`
- remove numbers
- remove punctuation and non-alpha noise
- drop media-only or effectively empty posts
- drop posts shorter than `min_text_length = 3`
- optionally drop non-English text

For the 32k BERT configs:

- English-only filtering: enabled
- augmentation: disabled

### 5.2 Determinism issue found during preprocessing

During the first BERT pass, language detection produced slightly different row counts across runs. This was traced to `langdetect` randomness.

The repo was later updated to make language detection deterministic with:

- `langdetect.DetectorFactory.seed = 0`

The preprocessing code was also hardened so symbol-only inputs do not crash language detection and instead fall back to a simple ASCII heuristic.

## 6. Classical Baseline Models

The baseline suite in `ragebait_detector/models/baselines.py` trained four scikit-learn models on two text views:

Text views:

- raw text
- cleaned text

Models:

- Logistic Regression
- linear SVC
- Gaussian Naive Bayes
- Decision Tree

Feature settings:

- TF-IDF
- `max_features = 20000`
- `ngram_range = (1, 2)`

### 6.1 Exact baseline comparison run

The cleanest apples-to-apples classical baseline run is:

- `outputs/bert_32k_tuned/20260417_200121/`

That run used the same tuned preprocessing and split sizes as the tuned BERT configuration, but ran **baselines only**.

Best exact-split classical baseline:

- Model: **raw-text linear SVC**
- Artifact: `outputs/bert_32k_tuned/20260417_200121/baselines/raw_svc_metrics.json`
- Accuracy: `0.8744`
- Rage-bait precision: `0.8306`
- Rage-bait recall: `0.8670`
- Rage-bait F1: `0.8484`

## 7. Initial BERT Baseline Attempt

### 7.1 Model architecture

The BERT classifier in `ragebait_detector/models/bert_classifier.py` used:

- Encoder: `bert-base-uncased`
- Classifier head:
  - pooled BERT output
  - dropout
  - linear layer
  - `GELU`
  - dropout
  - final linear layer to one logit

### 7.2 Initial config used

Initial run config: `configs/bert_32k.yaml`

Key settings:

- Model: `bert-base-uncased`
- Hidden dim: `128`
- Dropout: `0.2`
- Max length: `128`
- Stride: `48`
- Batch size: `8`
- Epochs: `1`
- Learning rate: `3e-5`
- Decision threshold: `0.5`

### 7.3 Initial run artifact

- Run directory: `outputs/bert_32k/20260417_192614/`

Observed preprocessing summary:

- Processed rows: `30,386`
- Dropped empty: `9`
- Dropped non-English: `1,605`

Observed split summary:

- Train: `24,310`
- Validation: `3,038`
- Test: `3,038`

### 7.4 Initial result

The first BERT run failed behaviorally. It effectively predicted the positive class too often.

Test metrics:

- Accuracy: `0.4055`
- Not-ragebait F1: `0.0000`
- Rage-bait precision: `0.4055`
- Rage-bait recall: `1.0000`
- Rage-bait F1: `0.5770`

Interpretation:

- the model was not learning a balanced decision boundary
- it had collapsed into near-constant positive prediction behavior

### 7.5 Diagnosis

The trainer was applying class correction in two places at once:

- `WeightedRandomSampler`
- `BCEWithLogitsLoss(pos_weight=...)`

For this already class-shaped training set, that combination over-pushed the model toward class `1`.

## 8. Tuning Pass

### 8.1 Code-level tuning changes

To fix the failed BERT run, the project was updated to:

- make `WeightedRandomSampler` optional
- make `pos_weight` optional
- make language detection deterministic

This added two training controls:

- `training.use_weighted_sampler`
- `training.use_pos_weight`

### 8.2 Tuned config used

Tuned run config: `configs/bert_32k_tuned.yaml`

Key settings:

- Model: `bert-base-uncased`
- Hidden dim: `256`
- Dropout: `0.2`
- Max length: `128`
- Stride: `48`
- Batch size: `8`
- Epochs: `3`
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Patience: `2`
- Decision threshold: `0.5`
- `use_weighted_sampler: false`
- `use_pos_weight: false`
- augmentation disabled
- English-only filtering enabled

### 8.3 Tuned run artifact

- Run directory: `outputs/bert_32k_tuned/20260417_194800/`

Preprocessing summary:

- Processed rows: `30,430`
- Dropped empty: `9`
- Dropped non-English: `1,561`

Split summary:

- Train: `24,346`
- Validation: `3,042`
- Test: `3,042`

Validation history:

| Epoch | Validation accuracy | Validation rage-bait F1 |
| --- | ---: | ---: |
| 1 | `0.8655` | `0.8454` |
| 2 | `0.8807` | `0.8565` |
| 3 | `0.8725` | `0.8307` |

The best checkpoint came from the best validation F1 state, which was reached before the final epoch.

## 9. Final Baseline Model Result

### 9.1 Final reported BERT baseline

The tuned BERT run is the project's official **Baseline Model**.

Artifact:

- `outputs/bert_32k_tuned/20260417_194800/bert/test_metrics.json`

Test metrics:

- Accuracy: `0.8767`
- Not-ragebait precision: `0.9173`
- Not-ragebait recall: `0.8712`
- Not-ragebait F1: `0.8937`
- Rage-bait precision: `0.8240`
- Rage-bait recall: `0.8848`
- Rage-bait F1: `0.8533`
- Macro F1: `0.8735`

### 9.2 Comparison against the exact classical baseline

Exact-split comparison:

| Model | Accuracy | Rage-bait F1 |
| --- | ---: | ---: |
| Raw SVC baseline | `0.8744` | `0.8484` |
| Tuned BERT baseline | `0.8767` | `0.8533` |

Interpretation:

- tuned BERT slightly outperformed the strongest classical baseline on the saved tuned split
- the improvement was real but modest
- the major win was not raw accuracy alone; it was recovering from the failed first BERT run and producing a stable classifier

## 10. Operational Notes After Training

After training, a local inference script was added:

- `scripts/test_trained_bert.py`

That script was later adjusted so ad hoc text tests default to **forced English** unless `--no-force-english` is passed. This was done because short slang-heavy English inputs were sometimes misdetected by `langdetect` and rejected before scoring.

This change affected **local testing behavior**, not the saved BERT checkpoint.

## 11. Known Weaknesses of the Baseline

These limitations should be stated explicitly any time this baseline is reported:

- Labels are weak labels from an LLM, not a human gold standard.
- The task is binary and specific to rage-bait, not general toxicity.
- The dataset mixes multiple sources with different styles and distributions.
- Train/validation/test splitting was stratified by label, not grouped by `author_id` or `source`.
- Language detection is brittle on very short text.
- Performance metrics describe agreement with the weak-labeled test split, not true real-world validity.

## 12. Baseline-to-Primary Transition

This completed binary project should now be treated as the **Baseline Model track**.

### Baseline Model

- Task: binary rage-bait detection
- Dataset source: mixed multi-source unified dataset
- Labels: weak labels from `Qwen/Qwen2.5-3B-Instruct-AWQ`
- Final training file: `data/labeled/balanced_32000_c95_r60_40.csv`
- Final model: tuned `bert-base-uncased` classifier
- Final artifact: `outputs/bert_32k_tuned/20260417_194800/`

### Primary Model

The next project should be treated as a separate track:

- Task: multi-class classification
- Label set:
  - `Normal`
  - `Profanity`
  - `Trolling`
  - `Derogatory`
  - `Hate Speech`
- Dataset strategy: one dataset only
- Labeling strategy: direct class labels rather than binary rage-bait weak labels

The primary model should not be described as an extension of the binary weak-label experiment. It should be described as a **new supervised project**, with the binary BERT system retained only as a comparison baseline and a record of what was learned.

## 13. Reconstructed Command Sequence

The exact full-history shell transcript is not preserved in the repo, but this is the artifact-backed command path that matches the saved outputs.

### 13.1 Unify raw data

Interactive entrypoint:

```bash
python3 scripts/interactive_import.py --config configs/default.yaml
```

Observed result:

- `data/unlabeled/unified_unlabeled_posts.csv`
- `data/interim/source_manifest.json`

### 13.2 Weak-label the unified dataset

Equivalent entrypoint:

```bash
python3 scripts/label_with_vllm.py --config configs/default.yaml \
  --input data/unlabeled/unified_unlabeled_posts.csv \
  --output data/labeled/vllm_all_qwen_ragebait_labels.csv
```

This full-file invocation is reconstructed from the fact that the unified dataset and the weak-labeled dataset both contain `507,682` rows.

### 13.3 Analyze the weak-labeled output

```bash
python3 scripts/analyze_labeled_csv.py \
  --input data/labeled/vllm_all_qwen_ragebait_labels.csv
```

### 13.4 Create the final balanced training set

```bash
python3 scripts/balance_labeled_csv.py \
  --input data/labeled/vllm_all_qwen_ragebait_labels.csv \
  --limit 32000 \
  --confidence-threshold 0.95 \
  --label-ratio 60/40
```

### 13.5 Train the first BERT run

```bash
./.venv/bin/python scripts/run_pipeline.py \
  --config configs/bert_32k.yaml \
  run --skip-baselines
```

Result:

- `outputs/bert_32k/20260417_192614/`

### 13.6 Run exact-split classical baselines for the tuned setup

```bash
./.venv/bin/python scripts/run_pipeline.py \
  --config configs/bert_32k_tuned.yaml \
  run --baselines-only
```

Result:

- `outputs/bert_32k_tuned/20260417_200121/`

### 13.7 Train the tuned BERT baseline

```bash
./.venv/bin/python scripts/run_pipeline.py \
  --config configs/bert_32k_tuned.yaml \
  run --skip-baselines
```

Result:

- `outputs/bert_32k_tuned/20260417_194800/`

## 14. Bottom Line

The project started with a mixed-source social-media corpus, weak-labeled that corpus with Qwen through vLLM, shaped a high-confidence binary training set, and then trained a `bert-base-uncased` classifier in two passes.

The first BERT pass failed because class balancing was over-applied. The second pass fixed that mistake and produced the final binary baseline:

- tuned BERT accuracy: `0.8767`
- tuned BERT rage-bait F1: `0.8533`

That model is the correct reference point for the next phase of the project. The next phase should be treated as a new multi-class experiment, not as a continuation of the weak-label binary setup.
