import pickle

# Charger le modèle
with open('model/toxic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Exemple d'utilisation du modèle
# Remplacez `data` par l'entrée que vous souhaitez tester avec le modèle
result = model.predict(data)
print(result)
