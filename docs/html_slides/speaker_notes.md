# Speaker Notes: Machine Learning Ragebait Detection

## 1. Machine Learning Ragebait Detection
Open by framing the project as both an engineering system and a scientific correction. The final result is not just that BERT performs well; it is that the team moved from weak labels to a more defensible human-labeled benchmark.

## 2. What is rage bait?
Give a plain-language definition first. Emphasize that the project is not trying to ban disagreement or detect every emotional post. It focuses on content designed to farm negative reactions.

## 3. The incentive problem
Walk through the loop: the platform sees interaction, not necessarily quality. Rage bait exploits that by using anger as fuel for visibility. This motivates an automated detector as a way to interrupt the feedback loop.

## 4. The rage bait detector
This slide transitions from motivation to the actual project. The detector is not just a classifier in isolation; it is a tool for reducing the payoff of negative engagement farming.

## 5. From weak labels to gold labels
The weak-label system was still valuable because it proved the engineering stack worked. But for a final claim, the project needed human labels. This is the hinge of the whole presentation.

## 6. Human-labeled abusive-language corpus
Point out the two task formulations. For the binary task, everything non-Normal becomes positive. For the multiclass task, the model must distinguish the five original categories.

## 7. Frozen 80/10/10 splits
This is the reproducibility slide. The frozen split lets us compare Logistic Regression, SVC, FFNN, and BERT without wondering if a better score came from a lucky data split.

## 8. Three levels of representation
Emphasize that the baselines are intentionally strong. The project is not comparing BERT to weak straw-man models; it is asking how much BERT adds beyond cheap lexical models.

## 9. Reproducible ML pipeline
Keep this slide brief. The important point is that the repo does not only produce numbers; it preserves the artifacts needed to understand where the numbers came from.

## 10. Binary detection results
The binary task is not trivial, but it is easier than multiclass because many abusive posts have obvious lexical cues. BERT still improves meaningfully over the strongest non-transformer baseline.

## 11. High recall without sacrificing precision
Use this to explain why both precision and recall matter. False negatives miss harmful content; false positives incorrectly flag normal or controversial conversation.

## 12. Five-class classification results
Stress that macro F1 is the key metric because it treats minority classes seriously. BERT improves because it can use context, not just word identity.

## 13. BERT improves every class
BERT's improvement is not only an aggregate metric artifact. It improves all five classes, especially Profanity, Derogatory, and Hate Speech, where intent matters most.

## 14. Remaining failures are boundary failures
The dominant error pair is Derogatory and Trolling. This supports the conclusion that the hardest failures are semantic and annotation-boundary failures, not simply missing vocabulary.

## 15. Hard examples explain the numbers
Avoid dwelling on offensive text in the presentation. The useful point is that the hard-error subset does not have a dramatically higher OOV rate, so the toughest mistakes are not just vocabulary holes.

## 16. Better accuracy costs more
The operational story is balanced: use TF-IDF when speed and simplicity matter, use BERT when category distinctions matter. The multiclass GPU run shows BERT is still practical for offline or batch use.

## 17. What we learned
Summarize in three claims. Classical baselines are essential, shallow neural models are not automatically better, and BERT earns its compute cost on the hardest semantic distinctions.

## 18. Improving the detector
End the technical story with practical next steps. The detector works, but the hardest categories require better context and clearer labels, not just a larger model.

## 19. Sources and artifacts
Use this as the closing slide for questions. If asked about validity, return to the pivot: the final claims are based on human labels and held-out evaluation.
