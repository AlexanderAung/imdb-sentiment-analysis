# IMDB Movie Review Sentiment Analysis
This project implements sentiment analysis on movie reviews from the IMDB dataset using TensorFlow and Keras. The sentiment analysis task involves classifying movie reviews as positive or negative based on their text content. The dataset consists of movie reviews along with their corresponding sentiment labels.

## Table of Contents

* Introduction
* Setup
& Data Preprocessing
* Model Architecture
* Training
* Evaluation

### Introduction <a name="introduction"></a>

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment conveyed by a piece of text. In this project, we focus on sentiment analysis of movie reviews, aiming to classify them as positive or negative. We utilize the IMDB dataset, which contains a large number of movie reviews labeled with their sentiment.

### Setup <a name="setup"></a>

Begin by importing the necessary libraries and downloading the IMDB dataset. The dataset is organized into training, validation, and test sets, each containing positive and negative reviews. We shuffle and split the training set to create a validation set for model evaluation during training.

### Data Preprocessing <a name="data-preprocessing"></a>

Text data preprocessing is performed using the TextVectorization layer provided by TensorFlow. We tokenize the text and convert it into multi-hot encoded vectors, considering bi-gram features. This processed data is used to train, validate, and test the sentiment analysis model.

### Model Architecture <a name="model-architecture"></a>

The sentiment analysis model consists of a simple neural network architecture implemented using Keras. It comprises an input layer followed by a dense hidden layer with ReLU activation and dropout regularization. The output layer utilizes a sigmoid activation function to produce a binary classification output indicating the sentiment of the input review.

### Training <a name="training"></a>

The model is trained using the binary 2-gram processed data. We compile the model with the RMSprop optimizer and binary cross-entropy loss function. During training, we employ a callback to save the best performing model based on validation accuracy. The training process is executed for a specified number of epochs.

### Evaluation <a name="evaluation"></a>

Once training is completed, the saved model is loaded for evaluation on the test set. We assess the model's performance by computing its accuracy on the test data.

This documentation provides an overview of the sentiment analysis code for IMDB movie reviews. By following the instructions provided, users can reproduce the sentiment analysis task, train their models, and evaluate them on the IMDB dataset.

### Note

Ensure TensorFlow and its dependencies are installed before running the code.
The dataset download link and preprocessing steps are included in the code for convenience. However, users may need to adjust paths or customize the data preprocessing pipeline as needed.
