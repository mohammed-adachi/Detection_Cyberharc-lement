from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

# Charger et nettoyer les données
df = pd.read_csv('./data/dataser.csv')
df['message'] = df['message'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Représentation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle
with open('./app/model/model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("Modèle entraîné et sauvegardé dans 'app/model/model.pkl'.")
