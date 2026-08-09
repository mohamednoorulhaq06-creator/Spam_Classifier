# Spam  Classifier

## 1. Project Overview

The Spam Email Classifier is a Machine Learning project that classifies text messages into two categories:

- Spam
- Ham (Not Spam)

The project uses Natural Language Processing (NLP) techniques to preprocess text messages and machine learning algorithms to classify them.

Three machine learning algorithms were trained and compared:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (SVM)

The Linear SVM model achieved the best overall performance.

---

## 2. Project Objective

The main objective of this project is to develop a machine learning model that can automatically identify whether a given message is spam or legitimate.

The project demonstrates the complete machine learning workflow:

- Data collection
- Data preprocessing
- Exploratory Data Analysis
- Text preprocessing
- Feature extraction
- Model training
- Hyperparameter tuning
- Model evaluation
- Model comparison
- New message prediction
- Model saving

---

## 3. Dataset

### Dataset Name

SMS Spam Collection Dataset

### Dataset Source

Kaggle / UCI

The dataset contains SMS messages labelled as either:

- `ham` - legitimate message
- `spam` - unwanted/spam message

### Original Dataset

- Total records: 5,572
- Columns used: 2
- Original columns: `v1`, `v2`

The columns were renamed to:

- `label`
- `message`

After removing duplicate records:

- Final records: 5,169

### Class Distribution

| Class | Number of Messages | Percentage |
| ----- | -----------------: | ---------: |
| Ham   |              4,516 |      87.4% |
| Spam  |                653 |      12.6% |

The dataset is therefore imbalanced, with significantly more ham messages than spam messages.

---

## 4. Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Joblib
- Jupyter Notebook

---

## 5. Project Structure

```text
Spam_Email_Classifier/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── Spam_Email_Classifier.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
├── src/
│
├── README.md
└── requirements.txt
```

---

## 6. Data Preprocessing

The following data preprocessing steps were performed:

1. Removed unnecessary columns from the original dataset.
2. Renamed columns for better readability.
3. Checked for missing values.
4. Removed duplicate records.
5. Checked the distribution of spam and ham messages.

No missing values were found in the final dataset.

A total of 403 duplicate records were removed.

---

## 7. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the dataset.

The analysis included:

- Spam vs Ham message distribution
- Percentage distribution
- Message length analysis
- Word count analysis

The dataset contains approximately:

- 87.4% Ham messages
- 12.6% Spam messages

This indicates that the dataset is imbalanced.

---

## 8. Natural Language Processing

Text preprocessing was performed before training the machine learning models.

The following techniques were used:

- Lowercase conversion
- URL removal
- Punctuation removal
- Number removal
- Tokenization
- Stopword removal
- Stemming

---

## 9. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text messages into numerical features.

A maximum of 5,000 features was used.

The TF-IDF vectorizer was fitted on the training data and then used to transform the test data.

---

## 10. Train-Test Split

The dataset was divided into:

- 80% Training data
- 20% Testing data

A random state of 42 was used for reproducibility.

Stratified splitting was used to maintain the class distribution between training and testing datasets.

---

## 11. Machine Learning Models

Three classification algorithms were trained and compared:

### Multinomial Naive Bayes

A probabilistic algorithm commonly used for text classification.

### Logistic Regression

A supervised classification algorithm used to predict the probability of class membership.

### Linear SVM

A Support Vector Machine that finds a decision boundary to separate the classes.

---

## 12. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics were selected because the dataset is imbalanced.

---

## 13. Final Model Performance

The best-performing model was **Linear SVM**.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 97.97% |
| Precision | 97.41% |
| Recall    | 86.26% |
| F1 Score  | 91.50% |

The Linear SVM achieved an overall accuracy of 97.97%.

The precision of 97.41% indicates that most messages predicted as spam were actually spam.

The recall of 86.26% indicates that the model detected most spam messages, but some spam messages were classified as ham.

The F1 Score of 91.50% provides a balance between precision and recall.

---

## 14. Confusion Matrix

The confusion matrix for the Linear SVM model was:

| Actual / Predicted | Ham | Spam |
| ------------------ | --: | ---: |
| Ham                | 900 |    3 |
| Spam               |  18 |  113 |

This means:

- 900 ham messages were correctly classified.
- 3 ham messages were incorrectly classified as spam.
- 113 spam messages were correctly classified.
- 18 spam messages were incorrectly classified as ham.

The confusion matrix image is available at:

```text
outputs/confusion_matrix.png
```

---

## 15. Model Comparison

The three models were compared using accuracy, precision, recall and F1 Score.

The comparison graph is available at:

```text
outputs/model_comparison.png
```

---

## 16. Hyperparameter Tuning

Hyperparameter tuning was performed on the **Linear SVM** model using `GridSearchCV` with **5-fold cross-validation**.

The `C` parameter was tested using the following values:

- `0.1`
- `1`
- `10`

The best parameter found was:

```text
C = 10
```

The best cross-validation F1 Score was:

```text
0.9002
```

### Tuned Linear SVM Performance

| Metric    | Score  |
| --------- | -----: |
| Accuracy  | 97.78% |
| Precision | 94.26% |
| Recall    | 87.79% |
| F1 Score  | 90.91% |

The tuned model improved recall from **86.26% to 87.79%**, meaning it detected a slightly higher proportion of spam messages.

However, the original Linear SVM achieved better overall test performance, with:

- Accuracy: **97.97%**
- Precision: **97.41%**
- F1 Score: **91.50%**

Therefore, the **original Linear SVM was retained as the final model**.

---

## 17. Model Saving

The trained Linear SVM model was saved using Joblib:

```text
models/spam_classifier.pkl
```

The TF-IDF vectorizer was also saved:

```text
models/tfidf_vectorizer.pkl
```

These files can be loaded later without retraining the model.

---

## 18. New Message Prediction

The trained model can classify new messages as either Spam or Ham.

Example:

```text
Congratulations! You have won a free iPhone. Click now to claim your prize!
```

Prediction:

```text
Spam
```

Example:

```text
Hi, can we meet tomorrow at 10 AM?
```

Prediction:

```text
Ham
```

---

## 19. Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 20. Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the following notebook:

```text
notebooks/Spam_Email_Classifier.ipynb
```

Run the notebook cells from beginning to end.

---

## 21. Requirements

The required Python packages are listed in:

```text
requirements.txt
```

Main dependencies include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- joblib
- jupyter

---

## 22. Future Improvements

Possible future improvements include:

- Testing additional machine learning algorithms
- Using larger spam datasets
- Using advanced NLP techniques
- Using word embeddings
- Using transformer-based NLP models
- Developing a web interface
- Creating an API for real-time spam detection

---

## 23. Conclusion

This project demonstrates a complete Natural Language Processing and Machine Learning pipeline for spam message classification.

Three machine learning algorithms were trained and evaluated using TF-IDF features.

Hyperparameter tuning was also performed on the Linear SVM model using GridSearchCV with 5-fold cross-validation.

Among the tested models, the original Linear SVM achieved the best overall test performance with:

**97.97% Accuracy and 91.50% F1 Score.**

The trained model can classify new messages as either Spam or Ham.
