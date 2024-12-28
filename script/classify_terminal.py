import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_curve,
    auc,
    accuracy_score,
)
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import label_binarize

# Charger et nettoyer les données
def load_and_clean_data(zfile_path):
    df = pd.read_csv(file_path)
    df['tweet_text'] = df['tweet_text'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
    return df

# Tracer une matrice de confusion
def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    disp.ax_.set_title('Matrice de confusion')
    disp.figure_.tight_layout()
    plt.show()

# Tracer les courbes ROC
def plot_roc_curves(y_test, y_pred_prob, class_names):
    y_test_bin = label_binarize(y_test, classes=class_names)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Baseline
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC par classe')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Entraîner un modèle
def train_and_evaluate_model(df, model_path):
    X = df['tweet_text']
    y = df['cyberbullying_type']  # Multicatégories
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Représentation TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Entraîner un modèle
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test_tfidf)
    y_pred_prob = model.predict_proba(X_test_tfidf)
    
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy:.2f}")
    
    loss = log_loss(y_test, y_pred_prob)
    print(f"Log-loss : {loss:.4f}")
    
    # Matrice de confusion
    plot_confusion_matrix(y_test, y_pred, model.classes_)
    
    # Courbes ROC
    plot_roc_curves(y_test, y_pred_prob, model.classes_)
    
    # Sauvegarder le modèle et le vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"Modèle sauvegardé dans {model_path}")

# Classifier un message saisi
def classify_message(model_path, message):
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)
    message_cleaned = ''.join([char for char in message.lower() if char.isalnum() or char.isspace()])
    message_vector = vectorizer.transform([message_cleaned])
    probabilities = model.predict_proba(message_vector)[0]
    predicted_class = model.classes_[probabilities.argmax()]
    predicted_prob = probabilities.max()
    return predicted_class, probabilities, predicted_prob, model

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
