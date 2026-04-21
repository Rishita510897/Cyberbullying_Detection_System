# Real-Time Cyberbullying Detection in User Generated Content Using NLP Techniques and SVM

## Project Overview

Cyberbullying has become a major issue on social media platforms such as Instagram, Twitter, Facebook, and online chat systems. Harmful messages, abusive comments, threats, and hate speech can seriously affect mental health, especially among students and teenagers.

This project aims to detect cyberbullying content in real time using Natural Language Processing (NLP) techniques and a Support Vector Machine (SVM) classifier.

The system analyzes user-generated text and predicts whether the message is:

- Cyberbullying
- Not Cyberbullying

A Streamlit web application is developed to provide a simple and user-friendly interface for real-time detection.

---

## Project Title

**Real-Time Cyberbullying Detection in User Generated Content Using NLP Techniques and SVM**

---

## Objectives

- To collect and use a labeled cyberbullying dataset
- To preprocess text using NLP techniques
- To extract important features using TF-IDF
- To train an SVM model for classification
- To perform sentiment analysis for better prediction
- To build a Streamlit web app for real-time detection
- To support both manual input and automated CSV upload

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TextBlob
- Joblib
- Streamlit
- Regex (re)

### Machine Learning
- Support Vector Machine (SVM)

### Feature Extraction
- TF-IDF Vectorizer

### Dataset Source
- Kaggle Cyberbullying Dataset

---

## Project Structure

```bash
Cyberbullying_Project/
│
├── app.py
├── train_model.py
├── test_model.py
├── data.csv
├── model_pickle.pkl
├── vectorizer_joblib.pkl
└── README.md