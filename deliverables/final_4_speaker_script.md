# Final Presentation Script

## Speaker 1

Slides 1 through 4.

We are presenting a ragebait detection project, but the main story is broader than a single classifier. The project started with a large weak-label pipeline and ended with a smaller but more defensible human-labeled benchmark. Our final source of truth is the completed CUDA metrics package in `deliverables/final_assets/complete_metrics/`.

On the title slide, the three numbers to remember are 12,490 gold-labeled posts, 0.9222 F1 for the final binary BERT detector, and 0.6405 macro F1 for the final five-class BERT model. The binary result is high because it asks a coarse question: normal or abusive/rage-inducing. The multiclass result is lower because it asks the model to separate Normal, Profanity, Trolling, Derogatory, and Hate Speech.

Ragebait is hard because the signal is not just offensive vocabulary. A post can be angry but legitimate, profane but not baiting, or mild on the surface but written to provoke a hostile reaction. The engagement loop matters here: inflammatory content gets posted, users react, ranking systems can treat that activity as relevance, and the content receives more visibility. Our detector is designed to identify that kind of content before users feed the loop.

The project pivot is important. In Iteration 1, we unified 507,682 posts and weak-labeled them with Qwen through vLLM. The best weak-label BERT model reached 0.8735 macro F1, but that score measured agreement with model-generated labels. It was useful engineering, not a final scientific endpoint. Iteration 2 moved to human labels, frozen splits, and a direct comparison among model families.

The final dataset has 12,490 rows. Normal has 5,053 examples, or 40.46 percent. Trolling has 4,537 examples, or 36.33 percent. Profanity has 1,582 examples, or 12.67 percent. Derogatory has 862 examples, or 6.90 percent. Hate Speech has 456 examples, or 3.65 percent. This class imbalance is why we care about macro F1 in the multiclass task.

## Speaker 2

Slides 5 through 8.

The same gold labels support two different tasks. In the binary task, Normal maps to 0 and every other label maps to 1. That task predicts whether the post should be treated as normal or abusive/rage-inducing. It is useful for screening and triage. In the multiclass task, the model predicts one of five labels. That task is harder but more informative because it tells us what kind of content the model thinks it is seeing.

Every final model uses the same frozen split. The training split has 9,992 rows, with 4,042 Normal and 5,950 abusive or ragebait-positive examples under the binary mapping. The validation split has 1,249 rows, with 506 Normal and 743 abusive examples. The test split also has 1,249 rows, with 505 Normal and 744 abusive examples. Validation is used for model selection and checkpoint selection; the test split is used only for final claims.

The preprocessing differs by model family, but the split does not. Logistic Regression and Linear SVC use TF-IDF with unigram and bigram features, a maximum vocabulary of 20,000 features, `min_df` of 2, `max_df` of 0.95, and sublinear term frequency. The FFNN uses a train-only vocabulary with lowercasing, max length 64, and a final vocabulary size of 10,975. BERT uses `bert-base-uncased` WordPiece tokenization with max length 128 in the final complete metrics run.

The model ladder is deliberately simple. Tier 1 is Logistic Regression and Linear SVC over sparse lexical features. Logistic Regression gives a probabilistic linear baseline, and Linear SVC gives a maximum-margin sparse text baseline. Tier 2 is a feed-forward neural network that learns token embeddings, mean-pools them, and applies hidden layers with ReLU and dropout. Tier 3 is BERT, which uses contextual transformer representations and a classifier head.

The multiclass task uses inverse-frequency class weights because the tail classes are small. The weights are 0.4944 for Normal, 1.5798 for Profanity, 0.5505 for Trolling, 2.8962 for Derogatory, and 5.4751 for Hate Speech. BERT uses AdamW, warmup and decay scheduling, gradient clipping, dynamic padding, and CUDA mixed precision where available. Binary BERT selects the best checkpoint by validation F1; multiclass BERT selects by validation macro F1.

## Speaker 3

Slides 9 through 12.

For binary classification, BERT is the best model, but the baselines are not weak. Logistic Regression reaches 0.8551 accuracy, 0.8371 precision, 0.9395 recall, and 0.8854 F1. Linear SVC reaches 0.8655 accuracy, 0.8780 precision, 0.8992 recall, and 0.8884 F1. The FFNN reaches 0.8655 accuracy, 0.8789 precision, 0.8978 recall, and 0.8883 F1.

The final binary BERT model reaches 0.9055 accuracy, 0.9054 precision, 0.9395 recall, and 0.9222 F1. Accuracy is the proportion of all test examples classified correctly. Precision tells us that when BERT predicts abusive or ragebait content, it is correct about 90.54 percent of the time. Recall tells us that it finds 93.95 percent of the true abusive or ragebait examples. F1 balances precision and recall, and BERT improves F1 by 0.0337 over the best non-transformer baseline.

The binary confusion matrix makes those metrics concrete. Out of 505 true Normal examples, BERT predicts 432 as Normal and 73 as abusive. Out of 744 true abusive examples, BERT catches 699 and misses 45. So the model's false-positive cost is higher than its false-negative count, but it is intentionally strong on recall, which matters if the use case is triage or moderation review.

The multiclass task is where BERT's contextual representation matters most. Logistic Regression reaches 0.6645 accuracy, 0.6645 micro F1, 0.5524 macro F1, and 0.6718 weighted F1. Linear SVC reaches 0.6797 accuracy, 0.6797 micro F1, 0.5403 macro F1, and 0.6764 weighted F1. The FFNN reaches 0.6517 accuracy, 0.6517 micro F1, 0.5250 macro F1, and 0.6606 weighted F1.

The final multiclass BERT model reaches 0.7390 accuracy, 0.7390 micro F1, 0.6405 macro F1, and 0.7441 weighted F1. Micro F1 equals accuracy here because each example has exactly one class. Macro F1 averages class F1 equally, so it is sensitive to minority classes. Weighted F1 weights classes by support, so it is closer to overall accuracy. BERT improves macro F1 by 0.0881 over the best non-transformer macro-F1 baseline.

Class-wise, BERT improves every label compared with Logistic Regression. Normal F1 improves from 0.8215 to 0.8772. Profanity improves from 0.6108 to 0.7349. Trolling improves from 0.6202 to 0.6853. Derogatory improves from 0.3398 to 0.4384. Hate Speech improves from 0.3696 to 0.4667. The tail classes remain hard, but BERT is consistently better.

The error analysis explains what remains difficult. The targeted hard-error file has 68 Trolling examples predicted as Derogatory, 25 Derogatory examples predicted as Trolling, and 2 Hate Speech examples predicted as Profanity. These are semantic boundary errors. The full test OOV rate is 7.83 percent, and the hard-error OOV rate is 7.61 percent, so unseen tokens are not the main explanation.

## Speaker 4

Slides 13 through 16.

The compute story is the practical tradeoff. TF-IDF models are extremely fast. Binary Linear SVC trains in 0.030 seconds and predicts the test split in 0.00014 seconds. Binary BERT trains in 173.3 seconds and predicts the test split in 3.411 seconds. That is a large compute increase, but it buys the best binary F1.

For multiclass, Logistic Regression trains in 7.4438 seconds, Linear SVC in 0.1864 seconds, and the FFNN in 23.6989 seconds. Multiclass BERT trains in 279.7 seconds, and the saved validation-plus-test prediction pass takes 7.575 seconds. The decision is therefore use-case dependent: TF-IDF is attractive for speed and simplicity, but BERT is justified when category distinctions matter.

There are important limitations. The model sees isolated posts, not full conversation threads. It does not know author history, target identity, or whether text is quoted, sarcastic, reclaimed, or part of a broader context. The split is stratified by label, not grouped by author or source, so a harder future evaluation should test grouped generalization. The minority classes are also small: the test split has only 86 Derogatory and 46 Hate Speech examples.

Ethically, this classifier should not be treated as a moderation policy by itself. False positives can suppress legitimate anger, satire, political speech, or marginalized language. The safer deployment role is assistive triage, user-facing friction, or review prioritization, with humans involved for high-impact decisions.

The final findings are straightforward. First, strong baselines matter; Logistic Regression and Linear SVC are hard to beat on binary text classification. Second, the FFNN does not add enough here because mean pooling loses too much sequence and pragmatic information. Third, BERT is the consistent winner, improving aggregate metrics and every class-wise F1 score.

The next path is not just more compute. The next version should test grouped splits by author or source, include reply-chain or conversation context, and refine label guidance around the Trolling, Derogatory, Profanity, and Hate Speech boundaries. All numbers in this presentation come from the final complete metrics package, and the references include the Trawling for Trolling dataset, BERT, scikit-learn, the Kaggle sources used in the archived weak-label corpus, and the project repository.
