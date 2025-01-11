import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
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


def train_and_evaluate_svm(df, model_path):
    X = df['headline']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("y_train",y_train)
    print("X_test_vec",X_test_vec)
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_vec, y_train)
    y_pred = svm_model.predict(X_test_vec)
    y_pred_prob = svm_model.predict_proba(X_test_vec)
    print("Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred))
    
    # Métriques globales (moyenne pondérée pour le multi-classe)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    
    print("\nMétriques globales :")
    print(f"Précision : {precision:.3f}")
    print(f"Rappel : {recall:.3f}")
    print(f"Score F1 : {f1:.3f}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Log-loss : {loss:.4f}")
    
    # Plot confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
    # disp.plot(cmap='viridis', xticks_rotation='vertical')
    # plt.title('Matrice de confusion - SVM')
    # plt.tight_layout()
    # plt.show()
    
    model_data = {
        'model': svm_model,
        'vectorizer': vectorizer
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Modèle sauvegardé dans {model_path}")
    return svm_model, vectorizer

    
def train_and_evaluate_model(df, model_path):
    X = df['headline']
    y = df['label']  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=100)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    y_pred_prob = model.predict_proba(X_test_tfidf)
    
    print("Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred))
    
    # Métriques globales (moyenne pondérée pour le multi-classe)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    
    print("\nMétriques globales :")
    print(f"Précision : {precision:.3f}")
    print(f"Rappel : {recall:.3f}")
    print(f"Score F1 : {f1:.3f}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Log-loss : {loss:.4f}")

    # plot_confusion_matrix(y_test, y_pred, model.classes_)
    
    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"Modèle sauvegardé dans {model_path}")
