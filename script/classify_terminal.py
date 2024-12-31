import pandas as pd
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
    # Initialize NLTK components
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Load data
    df = pd.read_csv(file_path)
    
    if 'tweet_text' not in df.columns:
        raise ValueError("Column 'tweet_text' is missing from the file.")
    
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
    df['tweet_text'] = df['tweet_text'].apply(clean_text)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['tweet_text'])
    
    # Tokenization
    df['tokens'] = df['tweet_text'].apply(word_tokenize)
    
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
    df['cleaned_text'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))
    
    # Remove empty rows
    df = df[df['cleaned_text'].str.strip().astype(bool)]
    
    # Handle missing values
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # Remove columns used for processing
    df = df.drop(['tokens', 'token_count'], axis=1)
    
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
    X = df['tweet_text']
    y = df['cyberbullying_type']  # Multicatégories
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    y_pred_prob = model.predict_proba(X_test_tfidf)
    
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy:.2f}")
    
    loss = log_loss(y_test, y_pred_prob)
    print(f"Log-loss : {loss:.4f}")
    
    # plot_confusion_matrix(y_test, y_pred, model.classes_)
    
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
