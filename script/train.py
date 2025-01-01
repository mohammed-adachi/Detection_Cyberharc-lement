# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import pickle
# import pandas as pd

# # Charger et nettoyer les données
# df = pd.read_csv('./data/cyberbullying_tweets.csv')
# df['tweet_text'] = df['tweet_text'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)

# # Préparer les labels : 1 pour cyberbullying, 0 pour not_cyberbullying
# df['label'] = (df['cyberbullying_type'] != 'not_cyberbullying').astype(int)

# # Représentation TF-IDF
# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(df['tweet_text'])
# y = df['label']

# # Diviser les données
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Entraîner un modèle
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Sauvegarder le modèle
# with open('./app/model/model.pkl', 'wb') as f:
#     pickle.dump((model, vectorizer), f)

# print("Modèle entraîné et sauvegardé dans 'app/model/model.pkl'.")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_curve,
    auc,
    accuracy_score,
)
import pickle
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords

# Télécharger les stopwords de NLTK
nltk.download('stopwords')

# Charger le modèle spaCy pour la lemmatisation
nlp = spacy.load("en_core_web_sm")

# Étape 1 : Collecte des données
df = pd.read_csv('./data/cyberbullying_tweets.csv')

# Étape 2 : Prétraitement des données
# Convertir en minuscules
df['tweet_text'] = df['tweet_text'].str.lower()

# Supprimer les caractères spéciaux et les chiffres
df['tweet_text'] = df['tweet_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Supprimer les mots vides (stopwords)
stop_words = set(stopwords.words('english'))
df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatisation avec spaCy
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if token.is_alpha])

df['tweet_text'] = df['tweet_text'].apply(lemmatize_text)

# Préparer les labels : 1 pour cyberbullying, 0 pour not_cyberbullying
df['label'] = (df['cyberbullying_type'] != 'not_cyberbullying').astype(int)

# Représentation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['tweet_text'])
y = df['label']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle et le vectoriseur TF-IDF dans un fichier pickle
with open('./app/model/model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("Modèle entraîné et sauvegardé dans 'app/model/model.pkl'.")
