from flask import Flask, request, jsonify
from pymongo import MongoClient
from flask_cors import CORS  # Importer CORS

# Créer une application Flask
app = Flask(__name__)

# Autoriser CORS pour toutes les routes et toutes les origines
CORS(app)

# Connexion à MongoDB (remplacez par l'URL de votre MongoDB si nécessaire)
client = MongoClient("mongodb://adachi:adachi@localhost:27017/")  # Connecte-toi à MongoDB localement
db = client["dbDetectCyberbullying"]  # Nom de la base de données
collection = db["messages"]  # Nom de la collection

# Route pour insérer un message
@app.route("/add_message", methods=["POST"])
def add_message():
    # Récupérer les données envoyées dans le corps de la requête
    data = request.get_json()

    # Vérifier que les champs nécessaires sont présents
    if "message" not in data or "cyberbullying_type" not in data:
        return jsonify({"error": "Les champs 'message' et 'cyberbullying_type' sont nécessaires."}), 400
    
    # Créer un document avec les données
    message_document = {
        "message": data["message"],
        "cyberbullying_type": data["cyberbullying_type"]
    }

    # Insérer le document dans la collection
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

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True)
