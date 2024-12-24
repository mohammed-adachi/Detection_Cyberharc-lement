import pandas as pd

# Charger le dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Aperçu des données :")
    print(df.head())
    return df

# Nettoyer les données (supprimer caractères spéciaux, etc.)
def preprocess_data(df):
    df['message'] = df['message'].str.lower()  # Convertir en minuscules
    df['message'] = df['message'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Supprimer caractères spéciaux
    return df

if __name__ == "__main__":
    file_path = "./data/dataser.csv"
    data = load_dataset(file_path)
    cleaned_data = preprocess_data(data)
    print("Données nettoyées :")
    print(cleaned_data.head())