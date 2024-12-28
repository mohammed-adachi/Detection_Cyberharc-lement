import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Charger et nettoyer les données
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    # Nettoyer les colonnes
    df['tweet_text'] = df['tweet_text'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
    return df

# Entraîner et sauvegarder le modèle
def train_and_save_model(df, model_path):
    # Préparer les features et labels
    X = df['tweet_text']
    y = df['cyberbullying_type']
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Représentation TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Entraîner un modèle avancé
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test_tfidf)
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarder le modèle et le vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"Modèle sauvegardé dans {model_path}")

# Main
if __name__ == "__main__":
    dataset_path = './data/cyberbullying_tweets.csv'  # Remplacez par le chemin de votre dataset
    model_path = './app/model/model.pkl'
    
    # Charger et nettoyer les données
    df = load_and_clean_data(dataset_path)
    
    # Entraîner et sauvegarder le modèle
    train_and_save_model(df, model_path)
