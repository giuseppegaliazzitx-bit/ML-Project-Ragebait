# Revision Rationale

## What Was Kept From Draft 1

Draft 1 had the strongest audience framing. The final version keeps its definition of ragebait, the negative engagement loop, the weak-label-to-gold-label project arc, and the emphasis that the core challenge is intent rather than profanity alone.

Draft 1 also had useful presentation pacing: motivation before methods, then dataset, model ladder, results, error analysis, compute, and future work. The final HTML deck preserves that broad progression while making the technical evidence denser and more consistent.

## What Was Reused From Draft 2

Draft 2 had the better technical structure. The final version reuses its evidence-first framing, binary versus multiclass comparison, model ladder, class-wise results concept, confusion-structure emphasis, and deployment tradeoff framing.

Draft 2's compact chart-driven style was also retained, but the final figures were regenerated from the completed metrics artifacts rather than copied from the draft.

## What Changed and Why

The final package replaces stale draft numbers with the authoritative CUDA complete metrics in `deliverables/final_assets/complete_metrics/complete_metrics.json`. For example, the final binary BERT test metrics are now reported as 0.9055 accuracy, 0.9054 precision, 0.9395 recall, and 0.9222 F1, and the binary confusion matrix is reconstructed from `binary_bert/bert_test_predictions.csv` as `[[432, 73], [45, 699]]`.

The final compute story was also corrected. The drafts still reflected older timing values in places, while the final version uses the complete metrics run: binary BERT training is 173.2670 seconds, multiclass BERT training is 279.6823 seconds, and multiclass BERT prediction timing is the saved validation-plus-test evaluation pass of 7.5751 seconds.

The final report is a new LaTeX report rather than a Markdown conversion. It adds academic structure, formal tables, generated figures, implementation details, explicit binary/multiclass framing, model mechanisms, error analysis, speed analysis, limitations, ethics, and references.

## Improvements in ML Depth

The final version explains how each tested final model works: Logistic Regression, Linear SVC, FFNN, and BERT. It also notes the archived Gaussian Naive Bayes and Decision Tree models as Iteration 1 weak-label baselines rather than final benchmark models.

The final version separates model mechanism from project implementation. It states the TF-IDF settings, FFNN vocabulary and mean-pooling design, BERT tokenizer and max length, optimizer choices, checkpoint criteria, class-weight formula, and frozen split policy.

The final binary and multiclass sections explain not only which model is better, but why the binary task is easier, why macro F1 matters for multiclass, and why BERT's largest value appears when label distinctions require context.

## Improvements in Evidence and Metric Consistency

Every reported final metric in the slides, report, notes, script, and charts was aligned to the complete metrics artifacts. The final charts under `deliverables/final_assets/` were regenerated directly from `complete_metrics.json`.

The final package avoids old approximate or conflicting values from the PowerPoint drafts. Binary BERT F1 is consistently reported as 0.9222, not the stale 0.9229 value. Binary BERT accuracy is consistently reported as 0.9055, not the stale 0.9079 value.

The final deck and report explicitly identify the metrics source so the audience can trace the numbers back to the CUDA run.

## Improvements in Structure and Visuals

The final HTML deck is not a PowerPoint export. It uses a consistent visual system, metric cards, generated charts, readable tables, and a stronger narrative flow across 16 slides.

The deck reduces generic overview content and replaces it with task framing, model details, metrics, confusion matrix interpretation, error analysis, and compute tradeoffs.

The final report and deck also share the same figure set, which keeps the package coherent and prevents slide/report drift.

## Citation and Artifact Coverage

All citation items appearing in the draft citation slides are included in both the final report and final slide deck: Trawling for Trolling, BERT, scikit-learn, the project repository, aadyasingh55's Twitter Emotion Classification Dataset, irakozekelly's Misleading Posts Driving Discourse on X, and speckledpingu's Raw Twitter Timelines dataset.

The final version also incorporates the completed metrics artifacts by using `complete_metrics.json`, BERT prediction CSVs, confusion matrices, training histories, and the multiclass hard-error CSV as the final evidence base.
