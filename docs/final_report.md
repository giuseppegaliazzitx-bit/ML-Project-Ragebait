# CS4375: Final Project Report - Ragebait Detection

**Repository:** https://github.com/giuseppegaliazzitx-bit/ML-Project-Ragebait

**Group 8:** Edison Cheah (ebc230001), Giuseppe Galiazzi (dal315218), Lawson Herger (lch220002), Andrew Lin (ael230000)

*Note:* This report follows the template's core structure, but it explicitly separates the repository into two phases: Iteration 1, the archived weak-label binary baseline, and Iteration 2, the final gold-label supervised system. That split is necessary because the repository's main scientific conclusion is not just "BERT worked well," but that weak-label agreement was insufficient and the project had to pivot to a better dataset and a better experimental design.

## 1 Introduction

The project studies how to detect rage-inducing or abusive online language from short social-media text. The repository ended up containing two related but scientifically different tasks. Iteration 1 built a binary rage-bait detector from a large mixed-source corpus that was weak-labeled by an instruction-tuned LLM. Iteration 2 replaced that track with a cleaner supervised benchmark built from `trolldata.csv`, a 12,490-row human-annotated dataset with five labels: `Normal`, `Profanity`, `Trolling`, `Derogatory`, and `Hate Speech`. The final system uses the human-labeled dataset as the authoritative evaluation surface.

The modeling strategy was a three-tier progression designed to test whether more expressive representations actually buy meaningful performance. Tier 1 uses TF-IDF with Logistic Regression and linear SVC as fast lexical baselines. Tier 2 uses a PyTorch feed-forward neural network (FFNN) over train-only token embeddings and mean pooling. Tier 3 fine-tunes `bert-base-uncased`, which can contextualize tokens and should therefore be better at separating subtle intent categories such as trolling, derogatory abuse, and implicit hate. Iteration 1 also included a separate weak-label pipeline based on vLLM and `Qwen/Qwen2.5-3B-Instruct-AWQ`, plus a deprecated manual-evaluation web app that helped motivate the final pivot away from noisy weak supervision.

The main results show a clear pattern. On the archived Iteration 1 weak-label task, a tuned BERT model reached 0.8767 test accuracy and 0.8735 macro F1, only slightly beating a strong raw-text linear SVC. On the final Iteration 2 human-labeled benchmark, BERT achieved the best held-out test performance on both tasks: 0.9079 accuracy and 0.9229 F1 for binary ragebait-vs-normal classification, and 0.7390 accuracy with 0.6405 macro F1 for the full five-class problem. The most persistent errors were not simple vocabulary misses. They came from category-boundary ambiguity, especially `Trolling <-> Derogatory`, plus cases where explicit profanity hid more severe intent such as threats or wishes of harm.

## 2 Data

### 2.1 Iteration 1: archived weak-label corpus

The first version of the project unified eight raw sources plus one unsupported file that was skipped during import. The final unified corpus was written to `data/unlabeled/unified_unlabeled_posts.csv`.

| Raw file | Source name | Rows imported |
| :--- | :--- | ---: |
| `AllTweets.csv` | `AllTweets` | 88,625 |
| `common_authors_data - general-anon.csv` | `common-annon` | 894 |
| `common_authors_data - israel-hamas-anon.csv` | `isreal-hamas-anon` | 434 |
| `common_authors_data - ukraine-anon.csv` | `ukraine-anon` | 202 |
| `common_authors_data - vaccine-anon.csv` | `vaccine-anon` | 145 |
| `sample_import.csv` | `sample_import` | 50 |
| `train-00000-of-00001.parquet` | `twtemotion` | 416,809 |
| `twitter_toxic_tweets.csv` | `twitter_toxic_tweets` | 31,962 |

After deduplication, the manifest records:

- Total rows written: 507,682
- Duplicate rows removed: 31,439
- Final schema: `post_id`, `author_id`, `created_at`, `language`, `text`, `source`

Weak labels were then generated for the full unified corpus. The saved analysis in `docs/iteration1_report.md` and the associated JSON artifacts show:

- Total labeled rows: 507,682
- Rage-bait positives: 31,072
- Overall positive rate: 6.12%
- High-confidence pool at `confidence >= 0.95`: 102,104 rows
- High-confidence positive rate: 12.68%

Because the raw weak-labeled set was too imbalanced for stable supervised training, the project created several balanced subsets. The final archived training file was `data/labeled/balanced_32000_c95_r60_40.csv`, which contained:

- 32,000 rows total
- 19,200 label-0 rows
- 12,800 label-1 rows
- Confidence range: 0.95 to 1.00

The repository also contains a deprecated exploratory manual-review path. The FastAPI/React manual-evaluation tool produced `data/labeled/manual_eval.csv` with 82 reviewed examples (69 label-0, 13 label-1). Its small size and unstable queue logic made it unsuitable as the final scientific dataset, but it helped confirm that the weak-label path needed to be replaced.

### 2.2 Iteration 2: final gold-label corpus

The final project direction uses `iteration2/data/raw/trolldata.csv`, a human-labeled abusive-language dataset with 12,490 examples. Unlike Iteration 1, the final dataset directly supervises the target task instead of asking a model to learn from another model's judgments.

The five-class label distribution is:

| Label | Rows | Share |
| :--- | ---: | ---: |
| `Normal` | 5,053 | 40.46% |
| `Profanity` | 1,582 | 12.67% |
| `Trolling` | 4,537 | 36.32% |
| `Derogatory` | 862 | 6.90% |
| `Hate Speech` | 456 | 3.65% |

From this dataset, the project defines two tasks:

1. Binary mapping:
   - `Normal -> 0`
   - all other labels -> `1`
2. Multiclass mapping:
   - `0 = Normal`
   - `1 = Profanity`
   - `2 = Trolling`
   - `3 = Derogatory`
   - `4 = Hate Speech`

Under the binary mapping, the class balance becomes:

| Binary label | Meaning | Rows | Share |
| :--- | :--- | ---: | ---: |
| `0` | Normal | 5,053 | 40.46% |
| `1` | Ragebait / abusive | 7,437 | 59.54% |

### 2.3 Canonical splits

All Iteration 2 experiments use the same stratified `80/10/10` split frozen in `iteration2/data/processed/`. The saved manifests record:

| Split | Rows | Binary label 0 | Binary label 1 |
| :--- | ---: | ---: | ---: |
| Train | 9,992 | 4,042 | 5,950 |
| Validation | 1,249 | 506 | 743 |
| Test | 1,249 | 505 | 744 |

For the five-class task, the exact split distributions are:

| Split | Normal | Profanity | Trolling | Derogatory | Hate Speech |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Train | 4,042 | 1,265 | 3,630 | 690 | 365 |
| Validation | 506 | 158 | 454 | 86 | 45 |
| Test | 505 | 159 | 453 | 86 | 46 |

This shared split policy matters. It ensures that every model family sees the same train, validation, and test rows, so measured gains are attributable to modeling choices rather than to split drift.

## 3 Methodology

### 3.1 Labeling philosophy and task definition

The repository's original binary rage-bait definition is documented in `docs/annotation_guidelines.md`. A positive label is reserved for posts intentionally engineered to provoke anger, mock expected reactions, or bait hostile engagement. That distinction matters because not all toxic, emotional, or controversial content is rage-bait. Genuine outrage, political commentary, or ordinary profanity can still be negative examples under the original binary definition.

Iteration 2 does not preserve that exact annotation ontology. Instead, it uses the external five-class gold dataset and maps all non-`Normal` classes into a binary positive class when running Experiment 1. This means the final binary task is best interpreted as a practical "normal vs. abusive/rage-inducing language" benchmark rather than a pure recovery of the original bespoke rage-bait definition.

### 3.2 Iteration 1 weak-label pipeline

Iteration 1 consisted of four stages:

1. Multi-source import and schema unification through `legacy_iteration1_scripts/interactive_import.py`
2. Weak labeling with `legacy_iteration1_scripts/label_with_vllm.py`
3. Confidence filtering plus class shaping with `legacy_iteration1_scripts/balance_labeled_csv.py`
4. Classical and BERT training in the archived `legacy_iteration1_ragebait_detector/` package

The vLLM stage used `Qwen/Qwen2.5-3B-Instruct-AWQ` at temperature 0.0 and requested JSON outputs containing `is_ragebait`, `confidence`, and `reason`. This pipeline was useful engineering practice, but it introduced the central scientific weakness of Iteration 1: evaluation measured agreement with weak labels, not agreement with humans.

### 3.3 Tier 1: lexical baselines

Tier 1 provides simple, fast baselines built with scikit-learn. Both tasks use a TF-IDF representation with:

- `max_features = 20000`
- `ngram_range = (1, 2)`
- `min_df = 2`
- `max_df = 0.95`
- `sublinear_tf = True`

The binary task trains Logistic Regression and linear SVC without explicit class weighting. The multiclass task computes inverse-frequency class weights from the frozen train split and injects them into both Tier 1 models. These models are intentionally strong lexical baselines: they are cheap to train, easy to interpret, and often surprisingly competitive when labels correlate strongly with distinctive phrases or slurs.

### 3.4 Tier 2: sequence FFNN

Tier 2 is a compact neural baseline implemented in PyTorch. It builds a train-only word vocabulary, maps each token to a learned embedding, averages the non-padding embeddings, and feeds the pooled vector through a multi-layer perceptron with ReLU and dropout. The main hyperparameters are:

- `max_vocab_size = 20000`
- `min_freq = 2`
- `max_length = 64`
- `embedding_dim = 128`
- hidden layers: `[128, 64]`
- dropout: `0.30`

This model is more expressive than TF-IDF plus a linear classifier because it can learn dense feature interactions. At the same time, it still has an obvious limitation: average pooling discards word order and long-distance context, so it may not be much better than a lexical baseline on fine-grained intent distinctions.

### 3.5 Tier 3: fine-tuned BERT

Tier 3 fine-tunes `bert-base-uncased` using a lightweight classification head over the pooled encoder output. The implementation uses:

- Hugging Face `AutoModel` and `AutoTokenizer`
- dynamic padding through `DataCollatorWithPadding`
- AdamW with weight-decay parameter grouping
- linear warmup/decay scheduling
- gradient clipping
- optional AMP when CUDA is available

The binary BERT run used artifact-backed settings of two epochs and `max_length = 80` according to the saved training history, even though the current YAML file in the repo now lists three epochs and `max_length = 128`. The multiclass BERT run used four epochs and `max_length = 128`.

### 3.6 Handling class imbalance

Multiclass imbalance is severe enough that a naive loss would overfit the majority classes. The repository therefore computes exact class weights from the train split using:

`weight_i = N / (K * count_i)`

The resulting weights were:

| Class | Weight |
| :--- | ---: |
| Normal | 0.4944 |
| Profanity | 1.5798 |
| Trolling | 0.5505 |
| Derogatory | 2.8962 |
| Hate Speech | 5.4751 |

These weights are passed to `CrossEntropyLoss` for the multiclass FFNN and multiclass BERT runs.

### 3.7 Evaluation and error analysis protocol

Binary experiments use accuracy, precision, recall, and F1. Multiclass experiments use accuracy, micro F1, macro F1, weighted F1, and full per-class precision/recall/F1. Confusion matrices are saved for every archived run.

The deepest qualitative analysis is reserved for the multiclass BERT model. The repository's `iteration2/src/evaluation/error_analysis.py` extracts high-confidence misclassifications, especially on the pairs:

- `Trolling <-> Derogatory`
- `Profanity <-> Hate Speech`

Those targeted diagnostics are important because overall accuracy alone hides whether the model is learning the real semantic difference between ordinary profanity, provocation, derogation, and hate.

## 4 Implementations

### 4.1 Repository organization and reproducibility

The repo is deliberately split into an archived legacy surface and a clean final workspace:

- `legacy_iteration1_*`: preserved historical code, outputs, and tests
- `iteration2/`: final data processing, training, evaluation, and outputs
- `docs/`: roadmap, draft reports, iteration summaries, and examples

The strongest implementation detail in Iteration 2 is that it saves the objects needed for reproducibility:

- split manifests
- label maps
- class-weight artifacts
- training histories
- tokenizer snapshots
- checkpoints
- confusion matrices
- JSON summaries

The main reproducibility weakness is not in the model code, but in artifact consistency. The multiclass BERT summary stores held-out test metrics under a `validation` key, and the binary BERT YAML no longer exactly matches the saved run history. In this report, artifact-backed values take precedence over current config text whenever they disagree.

### 4.2 Dataset creation and label mapping

The Iteration 2 split generator is intentionally simple: it validates the required columns, inserts a stable `row_id`, maps labels, and freezes one stratified split for every downstream model.

```python
if task_type == "binary":
    label_map = _build_binary_label_map(dataframe, config)
    dataframe[target_column] = dataframe[label_column].map(label_map).astype(int)
    dataframe[target_name_column] = dataframe[target_column].map(
        {0: normal_label, 1: positive_label_name}
    )
```

*Listing 1: The binary label-mapping step in `iteration2/src/data/make_dataset.py`.*

This implementation choice is important because it guarantees that the binary and multiclass experiments are linked by the same underlying rows rather than by separately sampled datasets.

### 4.3 FFNN architecture

The FFNN keeps the sequence model intentionally lightweight. It learns token embeddings, masks out padding, mean-pools the remaining vectors, and classifies the pooled representation.

```python
embeddings = self.embedding(input_ids)
mask = (input_ids != self.pad_index).unsqueeze(-1)
masked_embeddings = embeddings * mask
pooled_embeddings = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
logits = self.classifier(pooled_embeddings)
```

*Listing 2: Mean-pooled FFNN forward pass from `iteration2/src/models/baselines.py`.*

This architecture is computationally cheap and stable, but it also explains why the FFNN stalls on the multiclass task: the average-pooled representation is better at capturing coarse lexical tone than fine-grained intent.

### 4.4 BERT training details

The BERT implementation follows standard fine-tuning practice:

- pretrained encoder: `bert-base-uncased`
- hidden and attention dropout: 0.1
- optimizer: AdamW with decay/non-decay parameter groups
- scheduler: linear warmup and decay
- clipping: `max_grad_norm = 1.0`
- checkpoint selection:
  - binary: best validation F1
  - multiclass: best validation macro F1

The code also enforces deterministic behavior by setting global seeds, disabling nondeterministic CUDA SDP kernels when applicable, and turning off CuDNN benchmarking. In practice, these controls make the saved split artifacts highly trustworthy.

### 4.5 Diagnostic and evaluation utilities

The repo includes evaluation helpers that do more than compute scalar metrics. They save confusion matrices, serialize JSON summaries, and programmatically extract difficult misclassifications for inspection.

```python
target_errors = misclassified[mask_troll_derog | mask_prof_hate]
target_errors = target_errors.sort_values(by="confidence", ascending=False)
target_errors[columns_to_save].to_csv(hard_errors_path, index=False)
```

*Listing 3: Hard-error extraction logic from `iteration2/src/evaluation/error_analysis.py`.*

This is one of the most useful parts of the final pipeline because it forces the report to analyze concrete linguistic failures instead of stopping at aggregate scores.

### 4.6 Auxiliary tooling

The repository also contains an abandoned manual-evaluation application with:

- FastAPI backend
- React/Vite frontend
- keyboard shortcuts for fast binary labeling
- a SQLite-backed queue state

Although this tool was not used for the final benchmark, it is still an implementation outcome worth recording because it shows the project explored human-in-the-loop labeling before settling on an external gold-standard dataset.

## 5 Experiments and Results

### 5.1 Experimental protocol and reporting caveats

All Iteration 2 models train on the frozen train split and use the frozen validation split for checkpoint selection or early stopping. The held-out test split is then used for final reporting.

However, the repository does not archive every result in the same way:

- the saved Tier 1 and Tier 2 summaries preserve validation metrics
- the multiclass BERT summary preserves held-out test metrics, but stores them under the field name `validation`
- the binary BERT summary preserves validation metrics only

To produce complete development-and-test tables for this report, I used the saved validation summaries exactly as archived and then reran the deterministic Tier 1 and Tier 2 models on the frozen test split inside the repository virtual environment. I also evaluated the saved BERT checkpoints on the same frozen test splits. The final tables below therefore reflect the repo's actual outcomes, but with cleaner reporting than the raw summaries provide on their own.

### 5.2 Iteration 1 historical baseline

The most important Iteration 1 outcomes are summarized below.

#### Weak-label dataset outcome

| Stage | Rows | Positive share |
| :--- | ---: | ---: |
| Unified unlabeled corpus | 507,682 | n/a |
| Weak-labeled corpus | 507,682 | 6.12% |
| High-confidence pool (`>= 0.95`) | 102,104 | 12.68% |
| Final balanced train file | 32,000 | 40.00% positive |

#### Archived model results

| Model / Run | Accuracy | Positive-class F1 | Macro F1 | Notes |
| :--- | ---: | ---: | ---: | :--- |
| Initial BERT (`bert_32k/20260417_192614`) | 0.4055 | 0.5770 | 0.2885 | Collapsed toward constant positive predictions |
| Raw-text linear SVC (`bert_32k_tuned/20260417_200121`) | 0.8744 | 0.8484 | 0.8706 | Strongest exact-split classical baseline |
| Tuned BERT (`bert_32k_tuned/20260417_194800`) | 0.8767 | 0.8533 | 0.8735 | Final archived baseline |

These results show two things. First, classical TF-IDF models were already strong enough that a transformer was never guaranteed to dominate. Second, the crucial Iteration 1 lesson was not the final score itself. It was that the whole evaluation still measured agreement with LLM labels. That realization is what made Iteration 2 necessary.

### 5.3 Experiment 1: Iteration 2 binary classification

#### Development results

| Model | Tier | Accuracy | Precision | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | 1 | 0.8695 | 0.8545 | 0.9408 | 0.8956 |
| Linear SVC | 1 | 0.8831 | 0.8912 | 0.9152 | 0.9031 |
| FFNN | 2 | 0.8855 | 0.8947 | 0.9152 | 0.9049 |
| BERT | 3 | 0.9111 | 0.9270 | 0.9233 | 0.9252 |

The validation table shows that the binary task is already fairly easy for lexical methods. Even the weakest baseline clears 0.895 F1, which means the positive class is strongly associated with recognizable lexical patterns. The FFNN only barely improves over linear SVC, suggesting that dense embeddings and nonlinear layers add little when the label boundary is mostly lexical.

#### Held-out test results

| Model | Tier | Accuracy | Precision | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | 1 | 0.8551 | 0.8371 | 0.9395 | 0.8854 |
| Linear SVC | 1 | 0.8655 | 0.8780 | 0.8992 | 0.8884 |
| FFNN | 2 | 0.8671 | 0.8734 | 0.9086 | 0.8906 |
| **BERT** | **3** | **0.9079** | **0.9210** | **0.9247** | **0.9229** |

On the held-out test set, BERT remains clearly best. Its 0.9229 F1 is 0.0322 above the best non-transformer baseline (FFNN at 0.8906). The binary task therefore validates the full modeling ladder: lexical models are strong, the FFNN is only marginally better, and contextual BERT still yields a real gain once the labels are human-grounded.

### 5.4 Experiment 2: Iteration 2 multiclass classification

#### Development results

| Model | Tier | Accuracy | Micro F1 | Macro F1 |
| :--- | :--- | ---: | ---: | ---: |
| Logistic Regression | 1 | 0.6869 | 0.6869 | 0.6044 |
| Linear SVC | 1 | 0.7038 | 0.7038 | 0.6027 |
| FFNN | 2 | 0.6950 | 0.6950 | 0.5994 |
| BERT | 3 | 0.7526 | 0.7526 | 0.6828 |

The validation results are much less forgiving than the binary task. Here, the FFNN fails to outperform the best Tier 1 baseline, which is strong evidence that average-pooled token embeddings are not enough to separate overlapping abuse categories. BERT, by contrast, opens a meaningful macro-F1 gap of about 0.078 over the strongest validation baseline.

#### Held-out test results

| Model | Tier | Accuracy | Micro F1 | Macro F1 |
| :--- | :--- | ---: | ---: | ---: |
| Logistic Regression | 1 | 0.6645 | 0.6645 | 0.5524 |
| Linear SVC | 1 | 0.6797 | 0.6797 | 0.5403 |
| FFNN | 2 | 0.6373 | 0.6373 | 0.5101 |
| **BERT** | **3** | **0.7390** | **0.7390** | **0.6405** |

Every model drops from validation to test on the five-class task, but BERT still preserves a large lead. Its test macro F1 of 0.6405 is 0.0881 above the best held-out baseline (Logistic Regression at 0.5524). This is the clearest evidence in the repo that contextual modeling matters once the task requires separating closely related abusive categories instead of merely detecting whether text is offensive at all.

#### Class-wise held-out comparison

The best non-transformer multiclass test model was Logistic Regression, so the most useful class-wise comparison is BERT vs. that baseline.

| Class | Logistic Regression F1 | BERT F1 |
| :--- | ---: | ---: |
| Normal | 0.8215 | 0.8772 |
| Profanity | 0.6108 | 0.7349 |
| Trolling | 0.6202 | 0.6853 |
| Derogatory | 0.3398 | 0.4384 |
| Hate Speech | 0.3696 | 0.4667 |

BERT improves every class, but the gains are not uniform. The strongest relative improvements are on `Profanity`, `Derogatory`, and `Hate Speech`, which is exactly where word identity alone is often insufficient.

### 5.5 Error analysis

#### Quantitative confusion structure

The multiclass BERT confusion matrix on the held-out test set reveals the dominant error pairs:

| Confusion pair | Symmetric error count |
| :--- | ---: |
| `Derogatory <-> Trolling` | 93 |
| `Normal <-> Trolling` | 85 |
| `Profanity <-> Trolling` | 66 |
| `Normal <-> Profanity` | 33 |
| `Derogatory <-> Hate Speech` | 29 |

The directional counts are even more revealing:

- `Trolling -> Derogatory`: 68
- `Trolling -> Profanity`: 54
- `Normal -> Trolling`: 45
- `Trolling -> Normal`: 40
- `Derogatory -> Trolling`: 25

So the model's biggest weakness is not confusion among all labels equally. It is specifically the instability of the `Trolling` boundary, which overlaps with mild profanity, direct insult, and some provocative but non-abusive discourse.

#### Qualitative failure modes

**1. Profanity masking more severe intent.**  
The highest-confidence targeted error in `hard_errors.csv` is:

- True label: `Hate Speech`
- Predicted: `Profanity`
- Confidence: 0.9632
- Text: `"get fucking cancer"`

The model recognizes explicit profanity, but it underestimates the severity of an indirect wish of harm. This is a semantic error, not a tokenization error.

**2. Trolling vs. derogatory blur.**  
Another high-confidence error is:

- True label: `Trolling`
- Predicted: `Derogatory`
- Confidence: 0.9470
- Text: `"Douchebag is a washed up has been who is mostly a crackhead bitch."`

This example contains direct insults, but the annotation marks it as trolling rather than pure derogation. The model appears to key on surface severity and collapses the pragmatic distinction.

**3. Derogatory content softened into trolling.**  
The reverse also appears:

- True label: `Derogatory`
- Predicted: `Trolling`
- Confidence: 0.9190
- Text: `"Stop even trying, we know that the only time you've been laid is when you got molested by your uncle."`

Here the model seems to over-read the taunting form and under-read the personal, targeted hostility.

**4. Long noisy rants dilute the label signal.**  
Several errors in `hard_errors.csv` are long, multi-sentence attacks with mixed rhetorical functions: insult, sarcasm, threat, complaint, and performative provocation. These examples often contain enough cues for several labels at once, which makes them inherently hard even for a contextual encoder.

#### Unknown-word analysis

The FFNN uses a train-only vocabulary of 10,975 tokens after frequency filtering. Measured at the token level:

- Overall multiclass test OOV rate: 7.83%
- Mean per-document OOV rate on the multiclass test set: 8.56%
- Hard-error subset OOV rate: 7.61%
- Mean per-document OOV rate on the hard-error subset: 8.74%

The key result is that the hard-error subset does **not** have a dramatically higher unseen-token rate than the overall test set. In other words, the worst BERT mistakes are not primarily caused by vocabulary holes.

The most frequent OOV tokens inside the hard-error subset were:

- `dickbutt`
- `dirtball`
- `nevar`
- `jabroni`
- `rancie`
- `friggin`
- `hola`

This still matters for the non-transformer models. The FFNN and TF-IDF baselines depend on exact surface forms, so creative spellings, usernames, and niche slang can weaken them. But the similar OOV rate between the full test set and the hardest BERT errors supports a stronger conclusion: the hardest remaining failures are semantic and annotation-boundary failures, not simply unknown-word failures. BERT's WordPiece tokenization largely avoids true unknown tokens, yet it still confuses `Trolling`, `Derogatory`, and milder forms of hate.

### 5.6 Speed analysis

#### Binary task timing

| Model | Train time (s) | Prediction time (s) | Approx. inference cost |
| :--- | ---: | ---: | ---: |
| Logistic Regression | 0.0472 | 0.00039 | 0.00031 ms/example on 1,249 validation rows |
| Linear SVC | 0.0323 | 0.00045 | 0.00036 ms/example on 1,249 validation rows |
| FFNN | 25.3877 | 0.01973 | 0.0158 ms/example on 1,249 validation rows |
| BERT | 4893.5949 | 64.8694 | 51.94 ms/example on 1,249 validation rows |

The binary BERT run happened on CPU, which made it by far the slowest model in the repository. That run still established a useful lesson: if the task is almost linearly separable, the marginal gain from BERT may not justify CPU-only training costs.

#### Multiclass task timing

| Model | Train time (s) | Prediction time (s) | Approx. inference cost |
| :--- | ---: | ---: | ---: |
| Logistic Regression | 16.3143 | 0.00097 | 0.00078 ms/example on 1,249 validation rows |
| Linear SVC | 0.4709 | 0.00146 | 0.00117 ms/example on 1,249 validation rows |
| FFNN | 87.8604 | 0.1501 | 0.1202 ms/example on 1,249 validation rows |
| BERT | 376.7855 | 9.8603 | about 3.95 ms/example across one validation pass plus one test pass (2,498 rows total) |

Unlike the binary run, the multiclass BERT run used CUDA. Even though the task was harder, GPU training reduced wall-clock time from roughly 81.6 minutes in the binary CPU run to roughly 6.3 minutes in the multiclass GPU run. Operationally, the speed story is therefore simple:

- TF-IDF models are effectively free to train and serve.
- The FFNN is still very cheap.
- BERT is the only model that pays a serious compute cost, but on GPU that cost is still practical for offline moderation or batch scoring.

### 5.7 Retrospective: weak labels vs. gold labels

The repository's most important outcome is the project pivot itself.

Iteration 1 demonstrated that the team could build a full ML stack:

- ingest heterogeneous raw data
- weak-label it with an LLM
- shape a trainable dataset
- train classical and transformer models
- save stable evaluation artifacts

But Iteration 1 also showed the ceiling of that approach. The best tuned BERT on weak labels reached 0.8767 accuracy and 0.8735 macro F1, yet those numbers still described agreement with `Qwen/Qwen2.5-3B-Instruct-AWQ`, not agreement with human judgment.

Iteration 2, by contrast, gives the project a much more defensible scientific story:

- one gold-labeled dataset
- one frozen split policy
- one three-tier model ladder
- one clear held-out benchmark

The final binary and multiclass BERT results are not just numerically better than the archived baseline in the aggregate. They are also conceptually stronger because they are grounded in human labels and accompanied by deeper error analysis.

## 6 Conclusion

This repository shows a complete machine-learning project lifecycle, including a failed direction, a corrective pivot, and a final defensible benchmark. Iteration 1 proved that the engineering stack worked, but it also exposed the central flaw of weak supervision: high scores against model-generated labels do not establish real validity. Iteration 2 fixed that problem by adopting a human-labeled dataset, freezing a canonical split, and evaluating a model ladder from TF-IDF baselines to a fine-tuned transformer.

The final conclusions are clear. First, classical lexical baselines are strong enough that they must always be reported, especially on the binary task. Second, a shallow FFNN offers only marginal benefit and can even underperform the best linear model on multiclass macro F1. Third, BERT is the only model in the repo that consistently improves both aggregate performance and class-wise robustness, reaching 0.9229 F1 on the binary test set and 0.6405 macro F1 on the five-class test set.

The remaining challenge is not simply more data or more compute. It is better handling of semantic overlap. The largest unresolved failures occur at the `Trolling`, `Derogatory`, and `Hate Speech` boundaries, especially when short profane expressions imply stronger hostility than their surface form alone suggests. The most promising next steps would therefore be grouped splits by author/source, richer context beyond a single post, and sharper annotation guidance for borderline abuse categories.

## References

[1] *Trawling for Trolling: A Dataset*. arXiv:2008.00525.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT, 2019.

[3] F. Pedregosa et al. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2011.
