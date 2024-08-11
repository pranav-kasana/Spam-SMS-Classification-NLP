# SMS Spam Classification Project

## Objective
The SMS Spam Classification project aims to develop a machine learning model capable of classifying SMS messages as either "spam" (unwanted messages) or "ham" (legitimate messages). This project utilizes natural language processing (NLP) techniques to preprocess the text data and employs various machine learning algorithms for classification.

## Table of Contents
- [Importing Libraries](#importing-libraries)
- [Loading the Dataset](#loading-the-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Exploring NaN Values](#exploring-nan-values)
  - [Class Distribution](#class-distribution)
- [Feature Engineering](#feature-engineering)
  - [Handling Imbalanced Datasets](#handling-imbalanced-datasets)
  - [Creating New Features](#creating-new-features)
- [Data Cleaning](#data-cleaning)
  - [Removing Special Characters and Numbers](#removing-special-characters-and-numbers)
  - [Converting Messages to Lowercase](#converting-messages-to-lowercase)
  - [Tokenization](#tokenization)
  - [Removing Stop Words](#removing-stop-words)
  - [Lemmatization](#lemmatization)
  - [Joining Lemmatized Words](#joining-lemmatized-words)
- [Building a Corpus](#building-a-corpus)
- [Model Building and Evaluation](#model-building-and-evaluation)
  - [Multinomial Naive Bayes](#multinomial-naive-bayes)
  - [Decision Tree](#decision-tree)
  - [Random Forest (Ensemble)](#random-forest-ensemble)
  - [Voting Classifier](#voting-classifier)
- [Making Predictions](#making-predictions)
- [Conclusion](#conclusion)

## Importing Libraries
To begin the project, essential Python libraries are imported, including:
- **NumPy**: For numerical operations and handling arrays.
- **Pandas**: For data manipulation and analysis, especially for handling data in DataFrame format.
- **Matplotlib and Seaborn**: For data visualization.
- **NLTK (Natural Language Toolkit)**: For natural language processing tasks, including tokenization and lemmatization.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Imbalanced-learn**: For handling imbalanced datasets.

## Loading the Dataset
The SMS Spam Collection dataset, a publicly available dataset containing 5,572 SMS messages labeled as either spam or ham, is loaded into a pandas DataFrame. The dataset consists of two columns:
- **v1**: Labels ("ham" for legitimate messages and "spam" for unwanted messages).
- **v2**: Actual SMS messages.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis is conducted to gain insights into the dataset's characteristics.

### Exploring NaN Values
The project checks for any missing values in the dataset using `isnull().sum()`. If any NaN values are found, appropriate measures are taken, such as dropping those rows or imputing values.

### Class Distribution
Countplots are generated using Seaborn's `countplot()` function to visualize the distribution of spam and ham messages, helping identify any class imbalance.

## Feature Engineering
Feature engineering involves creating new features or transforming existing ones to improve model performance.

### Handling Imbalanced Datasets
To address class imbalance, techniques such as oversampling the minority class (spam) are applied. The project employs the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to generate synthetic samples for the spam class.

### Creating New Features
New features are created based on the content of the SMS messages, such as `word_count`, which calculates the number of words in each message.

## Data Cleaning
Data cleaning is essential for preparing the SMS messages for analysis and modeling.

### Removing Special Characters and Numbers
Special characters and numbers are removed from the SMS messages using regular expressions.

### Converting Messages to Lowercase
All SMS messages are converted to lowercase to maintain consistency.

### Tokenization
The cleaned SMS messages are tokenized into individual words using NLTK's `word_tokenize()` function.

### Removing Stop Words
Stop words are removed using NLTK's stopwords corpus to reduce noise in the data.

### Lemmatization
Lemmatization is applied to convert words to their base or root form using NLTK's `WordNetLemmatizer`.

### Joining Lemmatized Words
After lemmatization, the lemmatized words are joined back into a single string, resulting in a cleaned and processed corpus of SMS messages.

## Building a Corpus
A corpus of cleaned and preprocessed messages is constructed, ready for model training.

## Model Building and Evaluation
The project builds and evaluates several machine learning models for SMS classification.

### Multinomial Naive Bayes
The Multinomial Naive Bayes model is trained on the preprocessed SMS messages, achieving an F1-score of 0.943.

### Decision Tree
A Decision Tree model is trained, achieving an F1-score of 0.98.

### Random Forest (Ensemble)
The Random Forest model achieves an impressive F1-score of 0.994.

### Voting Classifier
A Voting Classifier combines the predictions of the Multinomial Naive Bayes and Random Forest models for improved performance.

## Making Predictions
Once the models are trained and evaluated, they can be utilized to make predictions on new SMS messages.

## Conclusion
The SMS Spam Classification project successfully demonstrates the application of NLP techniques and machine learning algorithms for text classification. The methodologies employed can serve as a foundation for further research and improvements in spam detection systems.

## License
This project is licensed under the MIT License.
