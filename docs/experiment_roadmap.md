# Experiment Roadmap

## Purpose

This document has two jobs:

1. preserve a concise, artifact-backed record of Iteration 1, the completed weak-label binary baseline
2. define Iteration 2, the project's final supervised direction built on gold human annotations in `data/raw/trolldata.csv`

Iteration 1 remains the historical baseline. Iteration 2 becomes the primary research and reporting track.

## Iteration 1: Historical Baseline

### Summary

Iteration 1 established the repo's first end-to-end ragebait classifier. The project unified a mixed-source social-media corpus, generated weak binary labels with `Qwen/Qwen2.5-3B-Instruct-AWQ` through vLLM, shaped a high-confidence balanced training set, and then trained classical baselines plus a tuned `bert-base-uncased` model.

Key artifacts from that completed track:

- Weak-label output: `data/labeled/vllm_all_qwen_ragebait_labels.csv`
- Final training file: `data/labeled/balanced_32000_c95_r60_40.csv`
- Final baseline run: `outputs/bert_32k_tuned/20260417_194800/`
- Strongest exact-split classical baseline: raw-text linear SVC in `outputs/bert_32k_tuned/20260417_200121/`

Final reported BERT baseline metrics:

- Accuracy: `0.8767`
- Macro F1: `0.8735`
- Rage-bait F1: `0.8533`

### What Iteration 1 proved

- The repo can support full text-classification workflows from data preparation to saved evaluation artifacts.
- Classical TF-IDF models are strong enough that they must remain part of future comparisons.
- Transformer fine-tuning can beat the strongest linear model, but only after careful training stabilization.

### Limitation that ends this track

Iteration 1 measured agreement with LLM-produced weak labels, not with human-grounded truth. It remains a valid engineering baseline, but it is not the final scientific or product-facing result for this project.

### Preservation policy

The current vLLM labeling scripts, balancing utilities, configs, and saved outputs will be preserved as read-only Iteration 1 artifacts. In practice, that means the existing weak-label path centered on `scripts/label_with_vllm.py`, `scripts/balance_labeled_csv.py`, and the saved `outputs/bert_32k*` runs will be archived safely in the repo rather than repurposed for the new work. Iteration 2 will use new scripts and configs designed specifically for the gold-labeled dataset.

## Iteration 2: Gold-Label Final Direction

### Objective

Iteration 2 replaces the weak-label training path with a single-dataset, gold-standard supervised benchmark built from `data/raw/trolldata.csv`. The goal is to measure model quality against human annotations and show clear ML progression from classical methods to shallow neural models to transformer fine-tuning.

### 1. Dataset and label schema

Primary dataset:

- File: `data/raw/trolldata.csv`
- Rows: `12,490`
- Schema: `text`, `label`
- Annotation source: human labels
- Label set:
  - `Normal`
  - `Profanity`
  - `Trolling`
  - `Derogatory`
  - `Hate Speech`

Observed label inventory in the current CSV:

| Label | Rows | Share |
| --- | ---: | ---: |
| `Normal` | `5,053` | `40.46%` |
| `Profanity` | `1,582` | `12.67%` |
| `Trolling` | `4,537` | `36.32%` |
| `Derogatory` | `862` | `6.90%` |
| `Hate Speech` | `456` | `3.65%` |

This dataset replaces the mixed-source weak-label corpus as the authoritative source for final experiments.

### 2. Repo and data transformation

Iteration 2 changes the repo in a controlled way:

- Keep Iteration 1 assets intact for historical comparison and reproducibility.
- Treat the vLLM labeling pipeline as archived legacy code, not as part of the new training path.
- Build new dataset-validation, split-generation, preprocessing, training, and evaluation scripts around `trolldata.csv`.
- Keep the raw CSV immutable and generate all derived artifacts into processed, split, and output directories.
- Save label maps, split manifests, configs, and run summaries alongside model artifacts so every reported result can be reproduced.

The practical repo transition is therefore additive, not destructive: legacy weak-label assets stay preserved, while new gold-label assets are created in parallel.

### 3. Canonical split protocol

All Iteration 2 experiments will use one shared row-level split generated once and reused everywhere.

- Split ratio: `80/10/10`
- Target sizes: `9,992` train / `1,249` validation / `1,249` test
- Split method: stratified on the original 5-class gold labels
- Reuse policy: the exact same row assignments will be used across all model tiers and across both experiments
- Randomness control: fixed seeds for split creation, training, and evaluation
- Leakage policy: any deduplication or data-quality filtering happens before the split is frozen

Stratifying on the 5-class labels first is important. It preserves the fine-grained class distribution for the multi-class task while also letting the binary task inherit the same exact examples after label remapping.

### 4. Three-tier comparison strategy

Each experiment will compare the same progression of model families:

| Tier | Model family | Role in the study |
| --- | --- | --- |
| **Tier 1** | TF-IDF + Logistic Regression / linear SVM | Fast, interpretable baseline and first sanity check |
| **Tier 2** | FFNN or 1D-CNN | Intermediate neural baseline that can model nonlinear text patterns |
| **Tier 3** | Fine-tuned `bert-base-uncased` | Highest-capacity model and expected final reference system |

Comparison rules:

- All tiers train on the same train split and are selected on the same validation split.
- The test split stays fully held out until final evaluation.
- Preprocessing differences must be minimal and documented so gains are attributable to model capacity rather than dataset drift.
- Reporting must include both performance and operational tradeoffs such as training time, inference cost, and implementation complexity.

### 5. Experiment 1: Binary classification

Task definition:

- `Normal -> 0`
- `Profanity -> 1`
- `Trolling -> 1`
- `Derogatory -> 1`
- `Hate Speech -> 1`

Derived binary class distribution from the current gold dataset:

| Binary label | Meaning | Rows | Share |
| --- | --- | ---: | ---: |
| `0` | Normal | `5,053` | `40.46%` |
| `1` | Ragebait | `7,437` | `59.54%` |

This is already close to the desired mathematically balanced `40/60` regime, so no synthetic balancing step should be required at the dataset level.

Experiment goal:

- Establish the new human-grounded binary benchmark for ragebait-versus-normal detection.
- Compare how much performance is gained as the project moves from sparse linear models to shallow neural models to transformer fine-tuning.

Evaluation:

- Accuracy
- Precision
- Recall
- F1-score

Expected interpretation:

- Tier 1 should provide a strong lexical baseline.
- Tier 2 should test whether local nonlinear patterns improve over TF-IDF.
- Tier 3 should determine whether contextual modeling yields a meaningful lift once labels are trustworthy.

### 6. Experiment 2: Multi-class classification

Task definition:

- `0 = Normal`
- `1 = Profanity`
- `2 = Trolling`
- `3 = Derogatory`
- `4 = Hate Speech`

Experiment goal:

- Determine whether the models can separate closely related but linguistically distinct abusive-language categories.
- Test whether transformer fine-tuning is materially better at handling subtle distinctions such as `Trolling` vs. `Derogatory`, or general profanity vs. targeted hate speech.

Evaluation:

- Macro F1
- Micro F1
- Class-wise precision
- Class-wise recall
- Strict `5x5` confusion matrix

Interpretation focus:

- Macro F1 matters because the dataset is class-imbalanced and the rarest classes are operationally important.
- The confusion matrix is the main diagnostic tool for category overlap.
- Class-wise metrics must be reported even if overall accuracy looks strong.

### 7. BERT-focused error analysis

The deepest analysis effort in Iteration 2 will be reserved for the multi-class BERT model, because it is the most informative system for understanding label-boundary failures.

The analysis must answer not only "how often is BERT wrong?" but also "what linguistic patterns cause it to be wrong?"

Required error-analysis outputs:

- Review the highest-frequency confusion pairs, especially `Trolling <-> Derogatory`, `Profanity <-> Hate Speech`, and `Profanity <-> Trolling`.
- Inspect representative false positives and false negatives from each class on the held-out test set.
- Group errors by linguistic failure mode, such as sarcasm, quoted insults, indirect targeting, slang, short-context ambiguity, or profanity without clear abusive intent.
- Separate likely model failures from likely annotation-boundary cases.
- Tie findings back to concrete next steps such as preprocessing changes, class weighting, data augmentation, or annotation-guideline refinement if needed.

This section is critical because the value of Iteration 2 is not only a better score. It is also a defensible explanation of which linguistic markers are easy, which are hard, and why the transformer still confuses certain categories.

### 8. Execution order

Iteration 2 should proceed in this order:

1. Validate `trolldata.csv` and freeze the canonical label map.
2. Generate one stratified `80/10/10` split manifest and reuse it everywhere.
3. Run Tier 1 models on the binary task.
4. Run Tier 2 models on the binary task.
5. Fine-tune BERT on the binary task.
6. Repeat the same tier progression on the 5-class task.
7. Perform detailed BERT error analysis on the multi-class test set.
8. Consolidate results into comparison tables, figures, and final conclusions.

Running the binary experiment first is deliberate. It provides a lower-risk sanity check for the new gold-label pipeline before the project takes on the harder multi-class task.

### 9. Reporting requirements

Every Iteration 2 run should save:

- the exact config used
- the split manifest or split identifier
- label-mapping metadata
- validation and test metrics
- per-example test predictions
- confusion matrices where applicable
- training history for neural models
- a short run summary describing the model tier, task, and main result

The final write-up should present:

- one comparison table for binary classification across all three tiers
- one comparison table for 5-class classification across all three tiers
- a dedicated error-analysis subsection for the multi-class BERT run
- a short retrospective comparing Iteration 2 gold-label results against the archived Iteration 1 weak-label baseline

## Decision Statement

Iteration 1 remains in the repo as the historical binary weak-label baseline. Iteration 2 is the project's final direction: a gold-standard supervised benchmark built on `trolldata.csv`, evaluated through a three-tier modeling ladder, and reported with both aggregate metrics and deep error analysis.
