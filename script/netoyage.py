import pandas as pd
import re,string
import emoji
import unicodedata
import num2words
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
def load_and_clean_data(file_path):
    
    stop_words = set(stopwords.words('english'))
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    df = pd.read_csv(file_path)
    print(f"Original shape: {len(df)}")
    print(df.head())
    
    if 'headline' not in df.columns:
        raise ValueError("Column headline is missing from the file.")
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = emoji.demojize(text)
        text = re.sub(r':', '', text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
      
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    df['headline'] = df['headline'].apply(clean_text)
    df = df.drop_duplicates(subset=['headline'])
    
    df ['headline'] = df['headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df ['headline'] = df['headline'].apply(lambda x: regex.sub('',x))
    print( df.head())
    tok_list = []  # Initialize stemmer
    for index, row in df.iterrows():
     word_data = row['headline']
     nltk_tokens = nltk.word_tokenize(word_data)
     final = ' '.join(stemmer.stem(w) for w in nltk_tokens)
     tok_list.append(final)

    df['headline'] = tok_list
    df['tokens'] = df['headline'].apply(word_tokenize)
    print(df.head())
    df['headline'] = df['tokens'].apply(lambda tokens: ''.join([word for word in tokens if not word.isdigit()]))
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

if __name__ == "__main__":
  tfIdf= load_and_clean_data('./data/cyberbullying_tweets.csv')
  print(  tfIdf)

