# Audience Q&A

**Q1. Why not use only the binary detector if its F1 score is much higher than the multiclass model?**

**A1.** The binary detector is better for coarse screening, but it collapses profanity, trolling, derogation, and hate speech into one positive class. The multiclass model is lower-scoring because it solves the harder diagnostic problem that tells us where the model and labels blur.

**Q2. Does BERT's improvement justify its much higher compute cost?**

**A2.** It depends on deployment: TF-IDF is more practical for very high-throughput filtering, while BERT is justified when the system needs better recall and better separation among overlapping abuse categories.

**Q3. What evidence suggests the remaining errors are semantic rather than just vocabulary failures?**

**A3.** The targeted hard-error OOV rate is 7.61%, close to the full test OOV rate of 7.83%. The hard-error pairs cluster around Trolling vs. Derogatory and Hate Speech vs. Profanity, which points to label-boundary ambiguity rather than simple unknown words.
