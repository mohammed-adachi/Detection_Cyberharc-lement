import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
import nltk
import re
import pickle
import emoji
import num2words
import unicodedata
from collections import Counter
import warnings
from sklearn.preprocessing import label_binarize
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_curve,
    auc,
    accuracy_score,
)

warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_and_clean_data(file_path):
    """
    Load and clean text data with advanced preprocessing steps.
    """
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Load data
    df = pd.read_csv(file_path)
    
    if 'tweet_text' not in df.columns:
        raise ValueError("Column 'tweet_text' is missing from the file.")
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text)
        text = emoji.demojize(text)
        text = re.sub(r':', '', text)
        text = re.sub(r'\d+', lambda m: num2words.num2words(int(m.group())), text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[^a-z\s]', ' ', text)
        return ' '.join(text.split())
    
    # Apply initial cleaning
    df['tweet_text'] = df['tweet_text'].apply(clean_text)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['tweet_text'])
    
    # Tokenization
    df['tokens'] = df['tweet_text'].apply(lambda x: [
        lemmatizer.lemmatize(stemmer.stem(word))
        for word in word_tokenize(x)
        if word not in stop_words and len(word) > 2
    ])
    
    # Remove stopwords (already handled in the lambda function above)

    # Apply stemming and lemmatization (already handled in the lambda function above)
    
    # Join tokens back into cleaned text
    df['cleaned_text'] = df['tokens'].apply(' '.join)
    
    # Filter out meaningless phrases (very short or very long)
    df = df[df['cleaned_text'].str.split().apply(len).between(3, 100)]
    
    # Remove empty rows (handled by the length check above)
    
    # Handle missing values (already handled by the length check above)
    
    # Remove columns used for processing
    df = df.drop(['tokens'], axis=1)
    
    print(f"Original shape: {len(df)}")
    print(f"After cleaning: {len(df)}")
    
    return df

# Tracer une matrice de confusion
# def plot_confusion_matrix(y_test, y_pred, class_names):
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#     disp.plot(cmap='viridis', xticks_rotation='vertical')
#     disp.ax_.set_title('Matrice de confusion')
#     disp.figure_.tight_layout()
#     plt.show()

# Tracer les courbes ROC
# def plot_roc_curves(y_test, y_pred_prob, class_names):
#     y_test_bin = label_binarize(y_test, classes=class_names)
#     plt.figure(figsize=(10, 8))
#     for i, class_name in enumerate(class_names):
#         fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--')  # Baseline
#     plt.xlabel('Taux de faux positifs')
#     plt.ylabel('Taux de vrais positifs')
#     plt.title('Courbes ROC par classe')
#     plt.legend(loc='lower right')
#     plt.tight_layout()
#     plt.show()

# Entraîner un modèle et évaluer

def train_and_evaluate_model(df, model_path):
    X = df['cleaned_text']
    y = df['cyberbullying_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    
    # Reduced parameter grid
    param_grid = {
        'tfidf__max_features': [5000],
        'tfidf__ngram_range': [(1, 2)],
        'tfidf__min_df': [2],
        'clf__C': [0.1, 1.0],
        'clf__max_iter': [400],
        'clf__class_weight': ['balanced']
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,  # Reduced from 5
        scoring='f1_weighted',
        n_jobs=1,  # Sequential processing
        verbose=1
    )
    
    print("\nPerforming grid search...")
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Log-loss: {loss:.4f}")
    
    with open(model_path, 'wb') as f:
        pickle.dump((best_model, best_model.named_steps['tfidf']), f)
# Classifier un message saisi
def classify_message(model_path, message):
    with open(model_path, 'rb') as f:
        pipeline, vectorizer = pickle.load(f)
        
    # Apply same cleaning as training data
    message = clean_text(message)  # You'll need to make clean_text function available
    message_vector = vectorizer.transform([message])
    probabilities = pipeline.predict_proba(message_vector)[0]
    predicted_class = pipeline.classes_[probabilities.argmax()]
    predicted_prob = probabilities.max()
    
    return predicted_class, probabilities, predicted_prob, pipeline

# Main
if __name__ == "__main__":
    dataset_path = './data/cyberbullying_tweets.csv'  # Remplacez par le chemin réel
    model_path = './app/model/model.pkl'
    
    df = load_and_clean_data(dataset_path)
    train_and_evaluate_model(df, model_path)
    
    print("Entrez un message pour classifier (tapez 'exit' pour quitter) :")
    while True:
        user_message = input("Votre message : ")
        if user_message.lower() == 'exit':
            print("Au revoir !")
            break
        predicted_class, probabilities, predicted_prob, model = classify_message(model_path, user_message)
        print(f"Classe prédite : {predicted_class}")
        print(f"Probabilités par classe : {dict(zip(model.classes_, probabilities))}")
        print(f"Probabilité maximale : {predicted_prob}")

