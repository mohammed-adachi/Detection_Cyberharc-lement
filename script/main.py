import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_text(text):
    """Preprocess the text data."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    return text

def load_and_clean_data(dataset_path):
    """Loads the dataset and performs basic cleaning."""
    try:
        df = pd.read_csv(dataset_path)
        if 'tweet_text' in df.columns and 'cyberbullying_type' in df.columns:
            df = df[['tweet_text', 'cyberbullying_type']]
        elif 'tweet' in df.columns and 'cyberbullying' in df.columns:
            df = df[['tweet', 'cyberbullying']]
        else:
            raise ValueError("Expected columns not found in the CSV file.")
        
        df = df.dropna()
        df.columns = ['tweet', 'cyberbullying']
        df['tweet'] = df['tweet'].apply(preprocess_text)
        
        # Modify the cyberbullying labels
        df['cyberbullying'] = df['cyberbullying'].apply(lambda x: 'not_cyberbullying' if x == 'not_cyberbullying' else 'other_cyberbullying')
        
        # Add some simple, non-offensive messages to the dataset
        simple_messages = pd.DataFrame({
            'tweet': ['hello', 'hi', 'good morning', 'how are you', 'nice to meet you', 'have a good day'],
            'cyberbullying': ['not_cyberbullying'] * 6
        })
        df = pd.concat([df, simple_messages], ignore_index=True)
        
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading or cleaning the data: {str(e)}")
        raise

def train_and_evaluate_model(df, model_path):
    """Trains and evaluates a Logistic Regression model with probability calibration."""
    try:
        X = df['tweet']
        y = df['cyberbullying']
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        base_model = LogisticRegression(class_weight='balanced', C=0.1)
        model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logging.info(f"Model accuracy: {accuracy:.4f}")
        logging.info(f"Classification report:\n{report}")
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, './app/model/vectorizer.joblib')
    except Exception as e:
        logging.error(f"An error occurred during model training or evaluation: {str(e)}")
        raise

def classify_message(model_path, user_message):
    """Classifies a single user message."""
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load('./app/model/vectorizer.joblib')
        user_message = preprocess_text(user_message)
        user_message_vec = vectorizer.transform([user_message])
        probabilities = model.predict_proba(user_message_vec)[0]
        predicted_class = model.classes_[np.argmax(probabilities)]
        predicted_prob = np.max(probabilities)
        return predicted_class, dict(zip(model.classes_, probabilities)), predicted_prob
    except Exception as e:
        logging.error(f"An error occurred during message classification: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        dataset_path = './data/cyberbullying_tweets.csv'
        model_path = './app/model/model.joblib'
        
        logging.info("Loading and cleaning data...")
        df = load_and_clean_data(dataset_path)
        
        logging.info("Training and evaluating model...")
        train_and_evaluate_model(df, model_path)
        
        logging.info("Model training complete. Enter a message to classify (type 'exit' to quit):")
        while True:
            user_message = input("Your message: ")
            if user_message.lower() == 'exit':
                logging.info("Exiting the program.")
                break
            predicted_class, probabilities, predicted_prob = classify_message(model_path, user_message)
            logging.info(f"Predicted class: {predicted_class}")
            logging.info(f"Class probabilities: {probabilities}")
            logging.info(f"Highest probability: {predicted_prob:.4f}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

