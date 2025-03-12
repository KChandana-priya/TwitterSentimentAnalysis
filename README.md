# Twitter Sentiment Analysis on Pfizer Vaccines

## Introduction
This project focuses on performing sentiment analysis on tweets related to Pfizer vaccines using machine learning techniques. The objective is to classify tweets into three sentiment categories:
- **Positive Sentiment**
- **Negative Sentiment**
- **Neutral Sentiment**

To achieve this, various machine learning classifiers are implemented and evaluated based on performance metrics.

## Dataset
The dataset used in this project is sourced from Kaggle and consists of tweets containing discussions about Pfizer vaccines. The dataset comprises:
- **Tweet ID**: Unique identifier for each tweet.
- **Text**: Content of the tweet.
- **Sentiment Label**: Predefined sentiment classes (Positive, Negative, Neutral).
- **Timestamp**: Date and time of the tweet.
- **User Metadata**: Additional user-related features (if available).

## Data Preprocessing
To improve model performance, the raw text data undergoes preprocessing using the following steps:

1. **Text Cleaning**: Removing URLs, mentions (@user), hashtags, special characters, numbers, and unnecessary whitespace.
2. **Tokenization**: Splitting text into individual words (tokens) for further processing.
3. **Stopword Removal**: Filtering out common words (e.g., "the", "and") using NLTK.
4. **Stemming & Lemmatization**: Reducing words to their root forms to normalize text.
5. **Vectorization**: Converting text into numerical features using:
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - **Word2Vec or FastText embeddings**
   - **BERT embeddings (if using transformer models)**

## Model Selection & Training
The following machine learning classifiers are trained and evaluated:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**
4. **Naïve Bayes Classifier**
5. **LSTM (Long Short-Term Memory) for deep learning-based approach**

Each model is trained using labeled data and optimized using hyperparameter tuning. The models are evaluated based on:
- **Accuracy**: Overall correctness of predictions.
- **Precision, Recall, F1-Score**: Performance per class.
- **Confusion Matrix**: Visualization of correct and incorrect classifications.

## Evaluation & Results
Model evaluation is performed using a test dataset, and the performance is analyzed using:
- **Confusion Matrix**: Breakdown of actual vs predicted sentiments.
- **ROC Curves & AUC Scores**: Performance analysis of classifiers.
- **Word Cloud Visualization**: Most frequent words in positive and negative tweets.

## Requirements
To run the project, install the required dependencies:
```bash
pip install pandas numpy nltk scikit-learn tensorflow keras matplotlib seaborn wordcloud
```

## Execution
To execute the sentiment analysis pipeline, run:
```python
python sentiment_analysis.py
```