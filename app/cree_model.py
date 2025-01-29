from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump

# Charger un jeu de données exemple (Iris dataset)
def charger_donnees():
    data = datasets.load_iris()
    X = data.data  # Caractéristiques
    y = data.target  # Étiquettes
    return X, y

# Créer et entraîner le modèle SVM
def creer_et_entrainer_modele(X, y):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialiser le modèle SVM
    model = SVC(kernel='linear', probability=True, random_state=42)
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")
    
    # Évaluer le modèle
    score = model.score(X_test, y_test)
    print(f"Précision sur les données de test : {score:.2f}")
    
    return model

# Sauvegarder le modèle
def sauvegarder_modele(model, nom_fichier):
    dump(model, nom_fichier)
    print(f"Modèle sauvegardé dans le fichier : {nom_fichier}")

# Exécution principale
if __name__ == "__main__":
    # Charger les données
    X, y = charger_donnees()
    
    # Créer et entraîner le modèle
    model = creer_et_entrainer_modele(X, y)
    
    # Sauvegarder le modèle
    dump(model, "app/model/svm_model.pkl")
