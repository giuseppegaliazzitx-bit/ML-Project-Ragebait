# CS6301: Special Topics in Computer Science - Deep Learning for NLP
## Contextual Toxicity Detection using BERT 

**Authors:** Ekaterina Lepekhina, Sanatan Shrivastava, Vishwesh Dave 
**Institution:** The University of Texas at Dallas (Est. 1969) 

---

### Introduction & Research Problem
The ability to distinguish hate speech from offensive language is a major obstacle for automatic hate speech detection on social media. It is necessary to take into account the significant qualitative variations between several categories of potentially offensive language. For instance, a tweet that uses potentially offensive slurs should not be treated the same as one that quotes rap songs utilizing such language. This distinction is frequently not made in existing work, causing various forms of abusive language to be muddled. 

To address this, the project focuses on detecting hate speech by analyzing a pre-trained BERT model with an additional neural network mounted on top. Additionally, the document references Twitter Safety data illustrating a consistent downward trend in true hateful language impressions when non-violating viral tweets are removed.

---

### Previous Work
For many NLP activities, 2018 was a turning moment. Significant technological advancements included models like Google's BERT, OpenAI's Generative Pre-trained Transformer (GPT), Embedding from Language Models (ELMO), and Universal Language Model Fine-Tuning (ULMFIT). The project analyzes Google's BERT model (Devlin et al.) because utilizing its pre-trained layers saves considerable computation and training costs.

| Authors | Neural Network Approach | Year |
| :--- | :--- | :--- |
| Djuric et al. | Two-step strategy: Continuous bag of words model and binary classifier | 2015 |
| Waseem et al. | Multi-task learning framework to handle diverse datasets | 2016 |
| Gambck et al. | CNN model trained on various embeddings (word, character n-grams) | 2018 |
| Various | ULMFIT, ELMO, GPT, Google's BERT model | 2018 |


---

### Dataset: Hate Speech and Offensive Language
* **Source:** The project utilizes the "Hate Speech and Offensive Language" dataset from Davidson et al..
* **Size:** The dataset contains 24,783 tweets labeled for toxicity detection.
* **Vocabulary:** The vocabulary size of this dataset is 48,312.
* **Attribute - Tweet Text:** The `tweet text` attribute contains the content of the tweet.
* **Attribute - Count:** The `count` attribute tracks the number of times the tweet has been retweeted.
* **Attributes - Labels:** The `hate_speech`, `offensive_language`, and `neither` attributes act as binary indicators for the presence of each respective category.
* **Label Behavior:** The annotations are not mutually exclusive, meaning a tweet can be labeled as containing both hate speech and offensive language.
* **Imbalance:** Hate speech is less common than ordinary offensive speech. Weight balancing was used to address this problem of class imbalance.

---

### Methodology & Data Preprocessing
The primary methodology included fine-tuning a pre-trained BERT model on the dataset.

* **Model Selection:** The `bert-base-uncased` pre-trained model was used for pretraining and fine-tuning.
* **Data Cleaning:** Unnecessary characters, numbers, punctuation marks, and URLs were removed.
* **Data Formatting:** All text was converted to lowercase to ensure data consistency.
* **Feature Reduction:** Irrelevant columns, such as retweet count and the individual counts for hate/offensive/neither, were dropped as they do not contribute to classification.
* **Tokenization:** The text was tokenized and mapped to word embeddings using BERT's tokenizer and embedding layer.
* **Augmentation:** During training, data augmentation methods like token shuffle were used to enhance performance.
* **Data Splitting:** The dataset was split into an 80:10:10 ratio for training, validation, and testing.
* **Training Set:** The 80% split resulted in 19,826 sets for training.
* **Validation Set:** The 10% split resulted in 2,478 sets for validation.
* **Testing Set:** The 10% split resulted in 2,479 sets for testing.
* **Evaluation:** Validation and test sets were used to assess the model's performance using accuracy, precision, recall, and F1 score.

---

### Implementation & Experimentation
The team experimented by taking different layers of the BERT model into consideration, ranging from the simple CLS token output up to 13 (12 + CLS) layers. They also tested various neural network sizes, batch sizes, and epochs to enhance the implementation. Three primary classification models were fine-tuned on top of the pre-trained BERT model:

1.  **TF Auto Model For Sequence Classification:** This model considers only the CLS token of BERT for classification.
2.  **Simple Artificial Neural Network (ANN):** This was the most accurate model implemented. It takes the hidden state output of all 13 layers of the pre-trained BERT model as input. It implements a 2-layer deep neural network architecture with 512 neurons, a ReLU activation function, a dropout layer, and a softmax activation function. Training utilized a binary cross-entropy loss function and Adam optimizer, while early stopping was implemented to prevent overfitting.
3.  **Simple Convolutional Neural Network (CNN):** This model takes the summation of the output of all layers of BERT and places 1 CNN layer on top of it.

---

### Results
* **Accuracy:** The overall Sparse Categorical Accuracy achieved for the ANN model was 87%.
* **F1 Scores:** After 100 epochs, the "offensive language" class achieved the highest F1 score, while the "hate speech" class achieved an F1 score of only 0.20. The F1 score for "hate speech" versus "neither" is noticeable for 100 epochs.
* **Precision and Recall (Epoch 100):** Class 1 achieved a precision of 0.94 and a recall of 0.74, while Class 2 achieved a precision of 0.51 and a recall of 0.78.
* **Classification Difficulty:** It is easier for the model to differentiate between "no offensive language and/or hate speech" and "some presence of them" than to differentiate directly between "offensive language" and "hate speech". The sample size difference between the "neither" and "hate speech" classes is less drastic than comparing either to the "offensive language" class.
* **Resource Utilization:** Combining the ANN with BERT required extra computations, impacting speed and resources. On free-tier cloud resources, 20 epochs took 1.5 hours, 40 epochs took 2.5 hours, and 100 epochs took 6 hours.