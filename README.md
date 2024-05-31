# SMS Spam Classifier

This project implements an SMS spam classifier using a Naive Bayes algorithm. It processes SMS messages to determine whether they are spam or not.

## Features

- **Model Training**: Train a Naive Bayes classifier using SMS spam datasets.
- **Prediction**: Classify SMS messages as spam or ham (not spam).
- **Pre-trained Model**: Includes a pre-trained model for immediate use.

## Files

- `model.pkl`: Pre-trained Naive Bayes model.
- `naive_bayes_classifier.py`: Script to train and test the classifier.
- `naivebayes_classifier.ipynb`: Jupyter notebook for interactive exploration and model training.
- `sms.py`: Script to classify SMS messages using the pre-trained model.
- `spam.csv`: Dataset containing labeled SMS messages for training and testing.
- `spam1.csv`: Additional dataset for training and testing.
- `vectorizer.pkl`: Pre-trained vectorizer for transforming SMS messages into feature vectors.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
