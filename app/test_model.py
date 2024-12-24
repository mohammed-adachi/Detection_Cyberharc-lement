import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Exemple de données
messages = ["You are stupid", "Hello, how are you?", "Nobody likes you", "Good work on the project!"]
labels = [1, 0, 1, 0]  # 1 = Cyberharcèlement, 0 = Non-cyberharcèlement

# Créer un modèle simple
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)
model = LogisticRegression()
model.fit(X, labels)

# Sauvegarder le modèle
with open('model/model.pkl', 'wb') as file:
    pickle.dump((model, vectorizer), file)

print("Modèle entraîné et sauvegardé dans 'model/model.pkl'")
