# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using the Sentiment140 dataset. It involves preprocessing tweets, converting text to numerical features using TF-IDF, and training a logistic regression classifier to detect whether a tweet is positive or negative.

## 📌 Features

- Downloads the Sentiment140 dataset using Kaggle API
- Cleans and preprocesses tweet data (stopword removal, stemming, etc.)
- Vectorizes text using TF-IDF
- Splits data into training and testing sets
- Trains a logistic regression model
- Evaluates the model's accuracy
- Saves and reloads the trained model for future predictions

## 📂 Dataset

- **Source**: [Kaggle - Sentiment140](https://www.kaggle.com/kazanova/sentiment140)
- The dataset contains 1.6 million tweets labeled as positive or negative sentiment.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Satishnaidu2633/Twitter-Sentiment-Analysis
   cd twitter-sentiment-analysis
   
## 🚀 Future Enhancements

- **Expand Sentiment Classes**: Extend the binary classification (positive/negative) to include neutral sentiment for a more granular analysis.
- **Advanced Models**: Experiment with more powerful models like Support Vector Machines (SVM), XGBoost, or deep learning models (LSTM, CNN).
- **Transformers Integration**: Implement transformer-based models such as BERT or RoBERTa for improved contextual understanding of tweets.
- **Hyperparameter Tuning**: Use tools like GridSearchCV or RandomizedSearchCV to fine-tune model parameters for better performance.
- **Real-time Tweet Analysis**: Integrate Twitter API to fetch live tweets and perform real-time sentiment prediction.
- **Data Visualization**: Add interactive visualizations (e.g., word clouds, sentiment trends) using libraries like Plotly or Seaborn.
- **Web Application**: Build a user-friendly interface using Streamlit, Flask, or Django for real-time predictions from user input.
- **Multilingual Support**: Incorporate support for tweets in multiple languages using translation APIs or multilingual models.
- **Handle Imbalanced Data**: Apply techniques like SMOTE or class weighting to handle class imbalance more effectively.
- **Noise Reduction**: Enhance text preprocessing by handling emojis, hashtags, mentions, and URLs more intelligently.
