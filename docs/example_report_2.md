# Sarcasm detection using machine learning algorithms 

## 1 Introduction (5pt) 
Zhifang Deng (zxd210002) Zhangqi Wang (zxw190018) Hua Xiao (hxx190000) Xingyun Xue (xxx210002) 

Sarcasm can be considered as a sophisticated type of verbal irony where people express the opposite of what they actually mean, creating a disparity between the literal and intended meanings of their statement[1]. It is frequently encountered on social media platforms, leading to confusion in determining the polarity of a sentence. This poses a significant obstacle for computational systems that rely on such data for tasks like sentiment analysis, opinion mining, author profiling, and detecting online harassment [2]. To achieve accurate predictions in the analysis of language-based data, it is essential to develop techniques for detecting sarcasm. This is necessary to avoid misinterpreting sarcastic statements as literal ones. Sarcasm detection involves a dual classification task, determining whether a given sentence is sarcastic or not. This is a challenging task not only for computers but also for humans. To achieve high-quality performance in this task, one needs to understand the context of the situation, the relevant culture, and in some cases, the specific individuals or issues involved. Code is available at here.

## 2 Task (5pt) 
Our goal in this project is to tackle the difficult problem of sarcasm detection on Twitter. We are trying to use supervised machine learning (ML) methods and natural language processing (NLP) tools to detect sarcasm in the iSarcasmEval dataset which was used for iSarcasmEval shared-task (Task 6 at SemEval 2022) [3]. In this task, we will utilize binary target labels, sarcastic and non-sarcastic sentences, to do pairwise sarcasm identification. These two sentences convey the same meaning, and the non-sarcastic one was rephrased from the sarcastic one by the same author. The sarcastic one will be determined.

## 3 Related Work 

### 3.1 Traditional Approaches for Sarcasm Detection 
Sarcasm detection is a challenging problem due to the nuances and subtleties of sarcasm. Some approaches to solve this problem is to use traditional machine learning methods such as Logistic Regression [4], Gaussian Naive Bayes [5], and Support Vector Machines [6]. These methods have been used successfully for binary classification tasks and thus have also been applied to sarcasm detection. The goal of sarcasm detection is to classify a given text as either sarcastic or non-sarcastic. These traditional methods have been used to train models on annotated datasets and have achieved good results on sarcasm detection tasks. However, their performance could be limited by the complexity of sarcasm and the need to understand the context of the text.

### 3.2 Fine-tuning Pre-trained Models 
Recently, there has been a shift towards using deep learning models for sarcasm detection, which are state-of-the-art models such as BERT [7] and GPT-2[8]. These models have been pre-trained on large amounts of data and have demonstrated strong performance on a variety of language tasks. Fine-tuning these models involves training the model further on the smaller and task-specific dataset, which can efficiently improve the performance of training [9]. Because of its transferability of knowledge learned during pre-training, fine-tuning pre-trained models usually have a better performance even if the sentence to evaluate doesn't has strong correlation to trainset. It has been shown that these models were very effective for a variety of natural language processing tasks, including sentiment analysis, text classification, and language modeling. They significantly reduced the need of large amount of task-specific training data and leaded to faster training times [10][11]. In the context of sarcasm detection, fine-tuning pre-trained models can also help capture the nuances and subtleties of sarcasm, which can be challenging for traditional methods.

## 4 Data (5pt) 

### 4.1 Dataset Description 
The dataset used for our task was obtained from the iSarcasmEval dataset [3]. It contains a total of 4,335 tweets in English, including 867 sarcastic tweets and 3,468 non-sarcastic tweets. The dataset also contains the rephrased non-sarcastic statement of the sarcastic tweets provided by authors to convey the same message without using sarcasm. Our intention was to use both the English sarcastic tweets and their corresponding rephrased statement for sarcasm detection in our task.

### 4.2 Sarcasm Labeling 
For each text in the dataset, its author provided the label specifying its sarcastic nature (sarcastic or non-sarcastic), as well as the rephrased text. The rephrased text was provided by the authors themselves to eliminate labelling proxies. The linguistic expert provided the label specifying the category of ironic speech that it reflects. The dataset is represented as a list of texts, with each text accompanied by a sarcasm label indicating whether or not it is sarcastic.

### 4.3 Data Preparation 
From the dataset, we used 200 pairs of data consisting of English sarcastic tweets and their corresponding non-sarcastic rewrites for the task. The compliant test set provided by the authors consisted of these 200 pairs of data.

### 4.4 Applicant Data 
The statistics for the English dataset used in this task are shown in Table 1.

| | total | sarcastic | non-sarcastic |
| :--- | :--- | :--- | :--- |
| dataset | 4,335 | 867 | 3,468 |
| compliant | 400 | 200 | 200 |


*Table 1: Statistics for the English training set, test sets for our task, as discussed in Section 2.* 

### 4.5 Data Preprocessing 
In order to prepare the dataset for analysis, we performed several data preprocessing using Python. Firstly, all of hashtags in the tweets were replaced with a special mark, as hashtags are often neutral and not indicative of detecting sarcasm. Secondly, all mentions and URLs in the tweets were replaced with a generic term, as mentions are not typically related to the sarcasm. Emojis were removed from the tweets, as they are often used for emphasis rather than sarcasm. These preprocessing procedures were conducted to simplify the text and remove extraneous information that could interfere with the sarcasm detection technique. Next, we utilized the preprocessed version of the data as the input of our sarcasm detection technique. The implementation of these preprocessing procedures can improve the quality of the dataset and ensure that the sarcasm detection technique identify sarcasm in the statement accurately.

## 5 Methodology (20pt) 
The methods for preprocessing the dataset, the implementation of models and the evaluation are described in this section.

### 5.1 Data Preprocessing 
The data preprocessing was conducted by cleaning and transforming it in a way that would be suitable for our machine learning algorithms. We started with a dataset of sarcastic and non-sarcastic tweets, which we preprocessed in two ways: (1) without preprocessing (2) with preprocessing.

### 5.2 Baseline 
Several baseline models were implemented and evaluated to compare with our proposed methods. Specifically, we considered Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors, Support Vector Machines, Decision Trees, and a random classifier. We implemented these models using scikit-learn, a popular Python library for machine learning.

* **LogisticRegression:** A linear classification algorithm that produces the probability of a binary response variable based on a set of features.
* **Gaussian Naive Bayes:** A probabilistic algorithm based on Bayes' theorem, which produces the conditional probability of each class given the input features, assuming that all features are independent and normally distributed.
* **K-Nearest Neighbors:** A non-parametric algorithm that classifies an observation based on the majority class of its k-nearest neighbors in the feature space.
* **Decision Tree:** A tree-based algorithm that partitions the feature space recursively, by selecting the feature that maximizes the information gain at each node.
* **Random:** An ensemble algorithm that constructs multiple decision trees, and aggregates their predictions to improve generalization performance.

**Configuration:** The Logistic Regression model was trained using the lbfgs solver and default hyperparameters. Gaussian Naive Bayes was trained using the GaussianNB class in scikit-learn. The number of neighbors for the k-nearest neighbors model was set to 5, and the default distance metric (Euclidean distance) was used. Support Vector Machines were trained using the SVC class in the scikit-learn library, with the scale option for the gamma parameter. Decision trees were trained using the Decision Tree Classifier class, with a maximum depth of 5. Finally, the random classifier predicted labels at random with equal probability for each class.

### 5.3 Models 
We used the preprocessed data for training and evaluated three different machine learning models: BERT, GPT-2, and Logistic Regression.

* **BERT:** BERT [7] can take into account the full context of a word within a sentence or even a document. We used BERT base cased model for the sequence classification and fine-tuned on the sarcasm detection task. The Transformers library from Hugging Face was used to train and evaluate the model. Bert Tokenizer was used to preprocess the data and encode it in a format suitable for the model.
* **GPT-2:** GPT-2 has 1.5 billion parameters, making it one of the largest and most powerful language models to date. We also used the GPT-2 model for sequence classification and fine-tuned on the sarcasm detection task. A library named "tweetsDISTILGPT2fi_v4" from Hugging Face and the corresponding tokenizer were used to train and evaluate the performance of model on the datasets.

In addition, we assessed the performance of a cutting-edge model such as GPT-3.5-turbo-0301 on our dataset. This model is a highly advanced language model that has displayed impressive performance on a range of natural language tasks.

### 5.4 Evaluation 
To evaluate the model performance, we primarily used accuracy metric which represents the proportion of correctly identified pairs of sarcastic and non-sarcastic texts. Other metrics also were considered, such as precision, recall, and Fl-score, if they are available. Confusion matrices was created to visualize comparison among true positives, true negatives, false positives, and false negatives. To ensure a fair comparison of the different models, we used the same train-test split for all three models and the same hyperparameters for BERT and GPT-2 models.

## 6 Implementation (15pt) 
In the baseline, we adapted following models from sklearn: Logistic Regression, KNeighbors Classifier, SVC, Decision Tree Classifier, GaussianNB. In our experiment, We use pretrained bert model BertModel with its corresponding tokenizer, and gpt-2 with its GPT2Tokenizer from hugging face. In additional to models, we use pytorch as main framework, numpy and pandas for statistic. A colab link is also available to check.

Some important steps are mentioned as follows. Firstly, we split the data equally and reverse one of either to avoid problems caused by unbalanced data.

```python
paired = data [data['rephrase'].notnull()] [['tweet', 'rephrase']]
part1 = pd.DataFrame(paired.head(paired.shape[0] // 2))
part1['sarcastic'] = 0
part2 = pd.DataFrame(paired.tail(paired.shape[0] - paired.shape[0] // 2))
part2.rename({'tweet': 'rephrase', 'rephrase': 'tweet'}, inplace=True, axis='columns')
part2['sarcastic'] = 1
paired_shuffled = shuffle(pd.concat([part1, part2]), random_state=seed)
```


Secondly, we preprocessed the datasets to avoid not meaningful words have bad impact on classification accuracy.

```python
class DataPreprocess:
    def __call__(self, tweet):
        tweet = self.replace_emojis(tweet)
        tweet = self.replace_hashtags(tweet)
        tweet = self.replace_link(tweet)
        tweet = self.replace_mentions(tweet)
        return tweet

preprocess = DataPreprocess()
```


Then we evaludate the accuracy or baseline before and after preproceessing.

```python
def task_baseline(data_df, X_train, X_col1_test, X_col2_test, y_train, y_test):
    # training
    classifiers = [
        LogisticRegression(solver='lbfgs', ...),
        GaussianNB(),
        KNeighborsClassifier(),
        SVC(gamma='scale', probability=True,...),
        DecisionTreeClassifier(random_state=seed)
    ]
    names, accuracies = [], []
    for clf in tqdm(classifiers):
        ...
    # random
    ...
    res_df = pd.DataFrame({'classifier': names, 'accuracy': accuracies})
```


Then we trained our advanced model on preprocessed data.

```python
bert_model = BertForSequenceClassification.from_pretrained('bert-base-cased').to(device)
bert_trainer = Trainer(...) # 8 epochs
bert_trainer.train()
bert_trainer.evaluate()
```


## 7 Experiments and Analysis (45pt) 

### 7.1 Experiment Setup 
Several experiments were conducted to evaluate the performance of proposed methods. We first split the dataset into a training set (80%) and a test set (20%). We used the training set to fine-tune the BERT and GPT-2 models and to train the logistic regression and support vector machine classifiers. To evaluate the effectiveness of our proposed methods, we performed the following experiments:
* **Baseline:** We used the logistic regression, Gaussian Naive Bayes, K-Nearest Neighbors, Support Vector Machine, Decision Tree, and random classifiers as baseline models. We trained these models on the training set and evaluated their performance on the test set.
* **Fine-tuned BERT and GPT-2 models:** We fine-tuned the pre-trained BERT and GPT-2 models on our sarcasm detection dataset and evaluated their performance on the test set.
* **Logistic regression and support vector machine classifiers with BERT embeddings:** We extracted the BERT embeddings for each tweet and used them as features to train the logistic regression and support vector machine classifiers. We evaluated their performance on the test set.

We used accuracy, F1 score, precision, and recall as evaluation metrics for all experiments.

### 7.2 Results and Analysis 
In this section, we present the results of our experiments on sarcasm detection in tweets. Firstly, the result of baseline without preprocess is as follows.

| Classifier | Accuracy |
| :--- | :--- |
| Logistic Regression | 0.729885 |
| GaussianNB | 0.442529 |
| KNeighbors Classifier | 0.091954 |
| SVC | 0.701149 |
| Decision Tree Classifier | 0.402299 |
| random | 0.500000 |


*Table 2: Test Results for Each Classifier without Preprocess* 

After preprocessing as mentioned in 4.5, the result of baseline is as follows.

| Classifier | Accuracy |
| :--- | :--- |
| Logistic Regression | 0.758621 |
| GaussianNB | 0.431034 |
| KNeighbors Classifier | 0.132184 |
| SVC | 0.770115 |
| Decision Tree Classifier | 0.459770 |
| random | 0.505747 |


*Table 3: Test Results for Each Classifier with Preprocess* 

The improvement in accuracy observed in the table suggests that preprocessing can work in a way for improving the quality and relevance of text data, and for enhancing the performance of models that use that data as input. We trained and evaluated four different models: BERT, GPT-2. Apart from these, we try to apply bert tokenizor to logistic regression and support vector machine (SVM). We compared the performance of our models using different amounts of data.

| Model | Acc | F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| BERT | 0.9195 | 0.9195 | 0.9302 | 0.9091 |
| GPT2 | 0.6321 | 0.5151 | 0.7727 | 0.3863 |
| Log-Reg | 0.8965 | 0.8888 | 0.9729 | 0.8181 |
| SVC | 0.8965 | 0.8915 | 0.9487 | 0.8409 |


*Table 4: Test Results for Each Approach* 

Our best-performing model is BERT, which achieves an accuracy of 0.9195, an Fl-score of 0.9195, a precision of 0.9302, and a recall of 0.9091. The fine-tuned GPT-2 model achieved the lowest performance compared with any other method, which is still competitive in baseline with and without preprocessing. The logistic regression model and SVM achieved an accuracy of about 0.8966, an F1-score of 0.8889, a precision of 0.9730, and a recall of 0.8182. The support vector machine (SVM) model also achieved an accuracy of 0.8966, an F1-score of 0.8916, a precision of 0.9487, and a recall of 0.8409.

When applying the BERT tokenizer to the SVM and logistic regression models, we observed an improvement in accuracy, which suggests that the use of BERT embeddings helped to capture more nuanced and context-dependent features in the text data, resulting in better performance. One possible reason for this improvement is that BERT embeddings are pre-trained on large amounts of text data and are designed to capture the relationships and context between words in a sentence. By using these embeddings as inputs to the SVM and logistic regression models, the models were able to take advantage of this pre-existing knowledge and extract more meaningful and relevant features from the text data. To summarize, our experiments demonstrate the effectiveness of transformer-based models and the importance of having sufficient training data for sarcasm detection in a text classification task.

## 8 Conclusion 
From the results, it is clear that fine-tuned BERT and the logistic regression model based on BERT embeddings performed the best in detecting sarcasm in paired tweets. This indicates that contextualized word embeddings and fine-tuning on task-specific data are effective in capturing the nuances bewteen paired tweets. The lower performance of the fine-tuned GPT2 model compared to fine-tuned BERT and the logistic regression model using embeddings from BERT suggests that GPT2 may not be as effective as BERT in capturing the nuances of sarcasm in paired tweets. This could be because GPT2 is a generative language model and may not be as well-suited as BERT for the task of sarcasm detection. To summarize, the results suggest that fine-tuning on downstream data and using contextualized word embeddings are effective for detecting sarcasm in paired tweets. However, more research is needed to determine the optimal model architecture and fine-tuning strategy for this task.

### 8.1 Other advanced model 
We also evaluated the raw performance of the GPT3.5-turbo model (without learning from training data) and obtained an accuracy of 86.83%, which is notably high. Furthermore, we attempted to fine-tune the Davinci and Ada models using the tools provided by OpenAI, using the same data as in this experiment. However, the performance of both models was only slightly better than random, which may indicates that further optimization or data preprocessing is needed before finetuning.

### 8.2 Limitations and Future Work 
One limitation of our study is that the BERT configuration we used is complex and cannot be standardized across different studies. While this allows for flexibility and fine-tuning for specific tasks, it also makes it difficult to compare the performance of different BERT models. Another limitation is that we cannot guarantee the performance of the GPT-2 model we used from the Hugging Face library. While we selected a pre-trained model that was fine-tuned on tweets, it is possible that other pre-trained models or fine-tuning methods could have produced better results.

## References 
[1] D. Wilson, "The pragmatics of verbal irony: Echo or pretence?" Lingua, vol. 116, no. 10, pp. 1722-1743, 2006.
[2] S. Rosenthal, A. Ritter, P. Nakov, and V. Stoyanov, "SemEval-2014 task 9: Sentiment analysis in Twitter," in Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), Dublin, Ireland: Association for Computational Linguistics, Aug. 2014, pp. 73-80. DOI: 10.3115/v1/S14-2009. [Online]. Available: https://aclanthology.org/S14-2009.
[3] I. Abu Farha, S. V. Oprea, S. Wilson, and W. Magdy, "SemEval-2022 task 6: ISarcasmEval, intended sarcasm detection in English and Arabic," in Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022), Seattle, United States: Association for Computational Linguistics, Jul. 2022, pp. 802-814. [Online]. Available: https://aclanthology.org/2022.semeval-1.111.
[4] C. D. Manning and A. McCallum, "Logistic regression for binary classification," Speech and Language Processing, IEEE Transactions on, vol. 14, no. 2, pp. 227-238, 2002.
[5] D. J. Cohn, "Naive bayes classifiers," Machine learning, vol. 28, no. 1, pp. 145-178, 1996.
[6] C. Cortes and V. Vapnik, "Support vector machines," Machine learning, vol. 32, no. 2, pp. 183-202, 1995.
[7] J. Devlin, M.-W. Chang, K.-H. Lee, A. Hoiberg, and A. Radford, "Bert: Pre-training of deep bidirectional transformers for language understanding," arXiv preprint arXiv:1807.03814, 2018.
[8] A. Radford, C. Wu, M. Karras, and R. Durbin, "Gpt-2: Generative pre-training of text to text," arXiv preprint arXiv:1901.04198, 2019.
[9] Y. Bengo, I. Sutskever, and G. Hinton, "Fine-tuning of pretrained models for language understanding," arXiv preprint arXiv:1702.04882, 2017.
[10] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: pre-training of deep bidirectional transformers for language understanding," CoRR, vol. abs/1810.04805, 2018.
[11] N. Houlsby, A. Giurgiu, S. Jastrzebski, et al., "Parameter-efficient transfer learning for NLP," CORR, vol. abs/1902.00751, 2019.