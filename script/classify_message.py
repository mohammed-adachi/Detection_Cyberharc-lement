import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import chi2
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
import re
import pickle
import emoji
import num2words
import unicodedata
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from sklearn.preprocessing import label_binarize
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import (precision_score,
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_curve,
    recall_score,
    auc,
    accuracy_score,
)

# Classifier un message saisi
def classify_message(model_path, message):
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)
         # Add this label mapping
    label_mapping = {
        -1.0: "other_cyberbullying",
        0.0: "not_cyberbullying"
    }
    message_cleaned = ''.join([char for char in message.lower() if char.isalnum() or char.isspace()])
    message_vector = vectorizer.transform([message_cleaned])
    probabilities = model.predict_proba(message_vector)[0]
    predicted_class = model.classes_[probabilities.argmax()]
    predicted_prob = probabilities.max()
    probabilities_mappede = {label_mapping[class_]: prob for class_, prob in zip(model.classes_, probabilities)}
    return  label_mapping[predicted_class], probabilities_mappede, predicted_prob, model

def classify_message_SVM(model_path, message):
    # Charger le modèle et le vectoriseur
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    vectorizer = model_data['vectorizer']

    label_mapping = {
        0: "not_cyberbullying",
        -1: "other_cyberbullying"
    }
    
    # Nettoyer le message
    message_cleaned = re.sub(r'[^a-z0-9\s]', '', message.lower())
    # Transformer le message en utilisant le vectoriseur
    message_vector = vectorizer.transform([message_cleaned])
    predicted_class = model.predict(message_vector)[0]
    probabilities = model.predict_proba(message_vector)[0]
    
    predicted_label = label_mapping.get(predicted_class, "unknown")
    probabilities_mapped ={label_mapping[class_]: prob for class_, prob in zip(model.classes_, probabilities)} 
    return predicted_label, probabilities_mapped, np.max(probabilities), model
