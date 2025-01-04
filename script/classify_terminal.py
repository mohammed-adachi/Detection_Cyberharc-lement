import pandas as pd
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

warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_and_clean_data(file_path):
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original shape: {len(df)}")
    print(df.head())
    
    if 'headline' not in df.columns:
        raise ValueError("Column headline is missing from the file.")
    
    def clean_text(text):
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Convert emojis to text
        text = emoji.demojize(text)
        text = re.sub(r':', '', text)  # Remove colons from emoji text
        # Convert numeric values to text
        def convert_numbers(match):
            try:
                return num2words.num2words(int(match.group()))
            except:
                return match.group()
        text = re.sub(r'\d+', convert_numbers, text)
        
        # Normalize characters (remove accents and special characters)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    # Apply initial cleaning
    df['headline'] = df['headline'].apply(clean_text)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['headline'])
    
    # Tokenization
    df['tokens'] = df['headline'].apply(word_tokenize)
    
    # Remove stopwords
    df['tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])
    # Apply stemming and lemmatization
    def stem_and_lemmatize(tokens):
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
        return lemmatized_tokens
    
    df['tokens'] = df['tokens'].apply(stem_and_lemmatize)
    
    # Filter out meaningless phrases (very short or very long)
    df['token_count'] = df['tokens'].apply(len)
    df = df[(df['token_count'] >= 3) & (df['token_count'] <= 100)]
    
    # Join tokens back into cleaned text
    df['headline'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
    
    # Remove empty rows
    df = df[df['headline'].str.strip().astype(bool)]
    
    # Handle missing values
    df['headline'] = df['headline'].fillna('')
    
    # Remove columns used for processing
    df = df.drop(['tokens', 'token_count'], axis=1)
    
    print(f"After cleaning: {len(df)}")
    print(df.head())
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
def train_and_evaluate_svm(df, model_path):
    # Séparation des caractéristiques (X) et des étiquettes (y)
    X = df['headline']
    y = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    # Vérifier les valeurs uniques dans la colonne 'label'
    print(df['label'].unique())
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialiser le vectoriseur
    Vectorizer = CountVectorizer()
    X_train_vec = Vectorizer.fit_transform(X_train.values)

    # Initialiser le modèle SVM
    svm_model = SVC(kernel='linear')  # Utilisation d'un noyau linéaire pour une classification binaire
    svm_model.fit(X_train_vec, y_train)  # Entraînement du modèle
    
    # Prédiction sur l'ensemble de test
    X_test_vec = Vectorizer.transform(X_test)  # Transformation de X_test avec le même vectoriseur
    y_predict = svm_model.predict(X_test_vec)  # Prédiction des étiquettes sur les données de test
    
    # Calcul et affichage de la précision
    accuracy = accuracy_score(y_test, y_predict)
    print("Précision SVM : ", accuracy)
    
def train_and_evaluate_model(df, model_path):
    X = df['headline']
    y = df['label']  # Multicatégories
    
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
    plot_confusion_matrix(y_test, y_pred, model.classes_)

    # plot_confusion_matrix(y_test, y_pred, model.classes_)
    
    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"Modèle sauvegardé dans {model_path}")

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
    probabilities_mapped = {label_mapping[class_]: prob for class_, prob in zip(model.classes_, probabilities)}
    return  label_mapping[predicted_class], probabilities_mapped, predicted_prob, model
def classify_message_SVM(model_path, message):
    # Charger le modèle et le vectoriseur
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
    probabilities_mapped = {label_mapping[class_]: prob for class_, prob in zip(model.classes_, probabilities)}
    return  label_mapping[predicted_class], probabilities_mapped, predicted_prob, model


if __name__ == "__main__":
    dataset_path = './data/cyberbullying_tweets.csv'  # Remplacez par le chemin réel
    
    model_path = './app/model/model.pkl'
    
    df = load_and_clean_data(dataset_path)
 #   train_and_evaluate_model(df, model_path)
    train_and_evaluate_svm(df, model_path)
    print("Entrez un message pour classifier (tapez 'exit' pour quitter) :")
    while True:
        user_message = input("Votre message : ")
        if user_message.lower() == 'exit':
           print("Au revoir !")
           break
    #    predicted_class, probabilities_mapped, predicted_prob, model = classify_message(model_path, user_message)
    #    predicted_class, probabilities_mapped ,predicted_prob,model= classify_message_SVM(model_path, user_message) 
        print(f"Classe prédite : {predicted_class}")
        print(f"Classe prédite : {predicted_prob}")
        print(f"Probabilités par classe : {probabilities_mapped}")
