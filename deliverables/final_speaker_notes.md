# Final Speaker Notes

## Slide 1: From weak labels to a gold benchmark
Open by framing the project as both an ML system and a scientific correction. The headline result is not simply that BERT performs best; it is that the final claim is now based on human-labeled data and complete CUDA metrics artifacts.

State the three anchor numbers: 12,490 gold-labeled posts, 0.9222 binary BERT F1, and 0.6405 multiclass BERT macro F1.

## Slide 2: Ragebait rewards reaction volume
Define ragebait as content engineered to provoke anger or outrage because reaction volume creates visibility. Emphasize that the classification challenge is intent and context, not just profanity detection.

Use the post-reaction-ranking-reward loop to explain why automated detection could be useful: it can identify content before users feed the engagement loop.

## Slide 3: Volume was not enough
Explain the pivot. Iteration 1 unified 507,682 posts and weak-labeled them with Qwen through vLLM, but the best 0.8735 macro F1 score measured agreement with those weak labels.

The important lesson is that high volume and a high model score did not prove human-grounded validity. Iteration 2 moved to a smaller 12,490-row human-labeled benchmark with frozen splits.

## Slide 4: Five labels, uneven support
Walk through the label distribution. Normal and Trolling dominate the dataset, while Derogatory and Hate Speech are much smaller.

This imbalance explains why macro F1 is central for the multiclass task. Accuracy alone can look acceptable while minority abuse categories remain weak.

## Slide 5: The same labels answer two different questions
Contrast the task definitions clearly. Binary classification maps Normal to 0 and every other class to 1, so it asks whether a post should be flagged at all.

Multiclass classification keeps all five labels, so it asks what type of content the post contains. That is more useful diagnostically but much harder because the categories overlap.

## Slide 6: One frozen split for every model
Explain that all models use the same stratified 80/10/10 split: 9,992 train, 1,249 validation, and 1,249 test rows. Validation is for model selection; test is for final claims.

Summarize preprocessing by model family: TF-IDF for Logistic Regression and Linear SVC, train-only token vocabulary for FFNN, and BERT WordPiece tokenization with max length 128 for the final BERT runs.

## Slide 7: Cheap lexical baselines before contextual modeling
Describe the model ladder as a controlled capacity comparison. Logistic Regression and Linear SVC test whether sparse lexical features are already enough; FFNN tests a shallow learned dense representation; BERT tests contextual language modeling.

Mention the archived Gaussian Naive Bayes and Decision Tree models only as historical Iteration 1 baselines. They were not part of the final complete-metrics benchmark.

## Slide 8: The final package is auditable
Point to the implementation and artifact discipline: split manifests, label maps, tokenizer snapshots, confusion matrices, histories, and JSON summaries are saved.

Explain the key training controls: multiclass class weights handle imbalance, BERT uses AdamW with warmup/decay and gradient clipping, and checkpoints are selected by validation F1 or macro F1.

## Slide 9: Binary results
Read the BERT row explicitly: 0.9055 accuracy, 0.9054 precision, 0.9395 recall, and 0.9222 F1. Compare that with the best non-transformer, Linear SVC at 0.8884 F1.

The practical interpretation is that binary detection is fairly lexical, so baselines are strong. BERT still adds a meaningful 0.0337 F1 gain while retaining high recall.

## Slide 10: Binary confusion matrix
Read the confusion matrix: 432 true Normal examples were predicted Normal, 73 Normal examples were false positives, 45 abusive examples were missed, and 699 abusive examples were caught.

Use those counts to explain the metrics. Precision is 699 divided by all 772 predicted abusive cases, giving 0.9054. Recall is 699 divided by the 744 true abusive cases, giving 0.9395.

## Slide 11: Multiclass results
Read the BERT row: 0.7390 accuracy, 0.7390 micro F1, 0.6405 macro F1, and 0.7441 weighted F1. Compare that with Logistic Regression at 0.5524 macro F1, the best non-transformer macro score.

Explain why the gain is larger here than in binary classification. The multiclass labels require context and target interpretation, so BERT's contextual representation matters more.

## Slide 12: Error analysis
Explain that BERT improves all classes, but the hard errors cluster at label boundaries. The targeted hard-error file contains 68 Trolling to Derogatory errors, 25 Derogatory to Trolling errors, and 2 Hate Speech to Profanity errors.

Discuss the OOV check: the full test OOV rate is 7.83%, while the hard-error OOV rate is 7.61%. That means the hardest remaining errors are semantic boundary problems, not primarily unseen-token problems.

## Slide 13: Compute tradeoff
Read the key runtime comparison. Binary BERT trains in 173.3 seconds and predicts the test split in 3.411 seconds, while binary Linear SVC trains in 0.030 seconds.

For multiclass, BERT trains in 279.7 seconds and the saved validation-plus-test prediction pass takes 7.575 seconds. The deployment takeaway is that TF-IDF is best for speed, while BERT is justified when semantic category quality matters.

## Slide 14: Limitations and ethics
Explain that the model sees isolated posts, not full conversations. Without thread context, target identity, or speaker intent, some boundary errors are inevitable.

Emphasize responsible use. A detector like this should support triage, friction, or review queues, not automatically punish users, because false positives can suppress legitimate anger, satire, or marginalized speech.

## Slide 15: Final findings
Summarize the three technical conclusions: strong baselines are necessary, the FFNN did not add enough beyond them, and BERT consistently performs best.

Close the research story by saying the remaining challenge is semantic overlap among Trolling, Derogatory, Profanity, and Hate Speech. The next step is not just more compute; it is better context, grouped generalization tests, and sharper labels.

## Slide 16: References
State that the final slides and report use the completed metrics artifacts as the source of truth. The references include the Trawling for Trolling dataset, BERT, scikit-learn, the Kaggle sources used in the archived weak-label corpus, and the project repository.

Use this slide to invite questions about the binary versus multiclass framing, the weak-label pivot, or the model tradeoff between BERT quality and TF-IDF speed.
