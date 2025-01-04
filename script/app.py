from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS  # Importer CORS
import classify_terminal
import joblib

# Créer une application Flask
app = Flask(__name__)

# Autoriser CORS pour toutes les routes et toutes les origines
CORS(app)
    
    
  
    
# Connexion à MongoDB (remplacez par l'URL de votre MongoDB si nécessaire)
client = MongoClient("mongodb://adachi:adachi@localhost:27017/")  # Connecte-toi à MongoDB localement
db = client["dbDetectCyberbullying"]  # Nom de la base de données
collection = db["messages"]
   # Supprimer les messages existants
 # Nom de la collection
dataset_path = './data/cyberbullying_tweets.csv'  # Remplacez par le chemin réel
model_path = './app/model/model.pkl'
model_paths='./app/model/svm_model.pkl'
#  #   train_and_evaluate_model(df, model_path)
#     svm_model, vectorizer =  train_and_evaluate_svm(df, model_path)

df = classify_terminal.load_and_clean_data(dataset_path)
classify_terminal.train_and_evaluate_model(df, model_path)
svm_model, vectorizer = classify_terminal.train_and_evaluate_svm(df, model_paths)
model = joblib.load(model_path)
models = joblib.load(model_paths)
# Route pour insérer un message
@app.route("/add_message", methods=["POST"])
def add_message():
    # Récupérer les données envoyées dans le corps de la requête
    data = request.get_json()

    # Vérifier que les champs nécessaires sont présents
    if "message" not in data:
        return jsonify({"error": "Le champ 'message' est nécessaire."}), 400

    # Classifier le message
    predicted_class, probabilities_mappede, predicted_prob, model = classify_terminal.classify_message(model_path, data["message"])
    predicted_label, probabilities_mapped, confidence, _= classify_terminal.classify_message_SVM(model_paths, data["message"])
    # Créer un document avec les données et la classe prédite
    message_document = {
        "message": data["message"],
        "results": {
            "Logistic Regression": {
                "type_de_cyberharcèlement": predicted_class,
                "probabilités_par_classe": probabilities_mappede,
                "probabilité_maximale": predicted_prob
            },
            "SVM": {
                "type_de_cyberharcèlement": predicted_label,
                "probabilités_par_classe": probabilities_mapped,
                "probabilité_maximale": confidence
            }
        }
    }

    # Insérer le document dans la collection MongoDB
    result = collection.insert_one(message_document)
    
    # Retourner un message de succès avec l'ID du message inséré
    return jsonify({"success": True, "inserted_id": str(result.inserted_id)}), 201

# Route pour obtenir tous les messages
@app.route("/messages", methods=["GET"])
def get_messages():
    messages = list(collection.find())
    
    # Convertir les résultats en format JSON
    for msg in messages:
        msg["_id"] = str(msg["_id"])  # Convertir l'ObjectId en chaîne de caractères
    
    return jsonify(messages)
@app.route("/message/<string:name_message>", methods=["GET"])
def get_message_by_name(name_message):
    # Chercher le message dans la collection MongoDB
    message = collection.find_one({"message": name_message})

    if not message:
        return jsonify({"error": f"Message '{name_message}' non trouvé."}), 404

    # Convertir ObjectId en chaîne pour JSON
    message["_id"] = str(message["_id"])

    # Retourner les détails du message
    return jsonify(message), 200
# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True)
