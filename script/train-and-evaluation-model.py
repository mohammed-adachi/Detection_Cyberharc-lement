import wandb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, log_loss
import pickle
import netoyage  # Assurez-vous que ce module est correct

# Définir la configuration du Sweep pour optimiser les hyperparamètres
sweep_configuration = {
    'method': 'grid',  # Recherche exhaustive des combinaisons d'hyperparamètres
    'name': 'svm-hyperparameter-tuning',
    'metric': {
        'name': 'accuracy',  # Nous voulons maximiser la précision
        'goal': 'maximize'
    },
    'parameters': {
        'kernel': {
            'values': ['linear', 'rbf']  # Teste les noyaux 'linear' et 'rbf'
        },
        'max_features': {
            'values': [5000, 10000]  # Nombre de caractéristiques maximales à utiliser pour le TF-IDF
        },
        'test_size': {
            'values': [0.2, 0.3]  # Taille de l'ensemble de test
        }
    }
}

# Créer le Sweep avec wandb
sweep_id = wandb.sweep(sweep_configuration, project="news-classification")

def train_and_evaluate_svm():
    # Initialisation de wandb avec les hyperparamètres du Sweep
    wandb.init(project="news-classification", name="svm-model", config={})
    config = wandb.config

    # Chargement et préparation des données
    dataset_path = './data/cyberbullying_tweets.csv'  # Remplacez par le chemin réel
    df = netoyage.load_and_clean_data(dataset_path)
    X = df['headline']
    y = df['label']

    print("Test Size:", config.test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=42)
    
    # Initialisation du vectorizer TF-IDF
    vectorizer = TfidfVectorizer(max_features=config.max_features,  ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Création du modèle SVM avec les paramètres du Sweep
    svm_model = SVC(kernel=config.kernel, probability=True, random_state=42)
    svm_model.fit(X_train_vec, y_train)
    
    # Prédictions
    y_pred = svm_model.predict(X_test_vec)
    y_pred_prob = svm_model.predict_proba(X_test_vec)
    
    # Rapport de classification
    print("Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred))
    
    # Calcul des métriques globales
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
    
    # Sauvegarde du modèle
    model_data = {
        'model': svm_model,
        'vectorizer': vectorizer
    }
    model_path = './app/model/svm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Modèle sauvegardé dans {model_path}")
    
    # Fin de l'expérience wandb
    wandb.finish()

# Fonction pour exécuter le Sweep et entraîner le modèle avec chaque combinaison d'hyperparamètres
def run_sweep():
    wandb.agent(sweep_id, function=train_and_evaluate_svm)

# Lancer le Sweep
if __name__ == "__main__":
    run_sweep()
