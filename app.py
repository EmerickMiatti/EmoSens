"""
API REST pour la détection d'émotions
======================================

API Flask simple pour utiliser le modèle via HTTP.

Installation:
    pip install flask flask-cors

Lancement:
    python app.py

Utilisation:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "I am happy!", "threshold": 0.5}'
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.dirname(__file__))

from src.predict import predict_emotions, get_emotion_names

# Créer l'application Flask
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Activer CORS pour permettre les requêtes cross-origin

# Charger le modèle au démarrage (une seule fois)
print("Chargement du modèle...")
try:
    # Le modèle se charge automatiquement au premier appel de predict_emotions
    # On fait un appel test pour le charger maintenant
    predict_emotions("test", threshold=0.9)
    print("Modèle chargé avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    sys.exit(1)


@app.route('/')
def home():
    """Page d'accueil avec interface web."""
    return app.send_static_file('index.html')


@app.route('/api')
def api_info():
    """Informations sur l'API."""
    return jsonify({
        "name": "Emotion Detection API",
        "version": "1.0.0",
        "description": "API de détection d'émotions avec BERT Small (28 émotions)",
        "endpoints": {
            "POST /predict": "Prédire les émotions d'un texte",
            "GET /emotions": "Liste des émotions détectables",
            "GET /health": "Vérifier l'état de l'API"
        },
        "example": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "text": "I am so happy and excited!",
                "threshold": 0.5
            }
        }
    })


@app.route('/health')
def health():
    """Endpoint de santé pour vérifier que l'API fonctionne."""
    return jsonify({
        "status": "healthy",
        "model": "bert_small_en_uncased",
        "ready": True
    })


@app.route('/emotions')
def emotions():
    """Retourne la liste des 28 émotions détectables."""
    return jsonify({
        "count": 28,
        "emotions": get_emotion_names()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit les émotions d'un texte.
    
    Body JSON:
        {
            "text": "I am happy!",
            "threshold": 0.5  (optionnel, défaut: 0.5)
        }
    
    Response JSON:
        {
            "text": "I am happy!",
            "threshold": 0.5,
            "emotions": [
                {"emotion": "joy", "probability": 91.2},
                {"emotion": "excitement", "probability": 82.5}
            ],
            "count": 2
        }
    """
    # Vérifier que la requête est en JSON
    if not request.is_json:
        return jsonify({
            "error": "Content-Type doit être application/json"
        }), 400
    
    data = request.get_json()
    
    # Vérifier que le texte est fourni
    if 'text' not in data:
        return jsonify({
            "error": "Le champ 'text' est requis"
        }), 400
    
    text = data['text']
    threshold = data.get('threshold', 0.5)
    
    # Valider le seuil
    if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
        return jsonify({
            "error": "Le seuil doit être entre 0 et 1"
        }), 400
    
    try:
        # Prédiction
        emotions_list, all_probs = predict_emotions(text, threshold=threshold)
        
        # Formater la réponse
        emotions_formatted = [
            {"emotion": emotion, "probability": float(prob) / 100.0}
            for emotion, prob in emotions_list
        ]
        
        return jsonify({
            "text": text,
            "threshold": threshold,
            "emotions": emotions_formatted,
            "count": len(emotions_formatted)
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Erreur lors de la prédiction: {str(e)}"
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Prédit les émotions pour plusieurs textes.
    
    Body JSON:
        {
            "texts": ["I am happy", "I am sad"],
            "threshold": 0.5  (optionnel)
        }
    
    Response JSON:
        {
            "results": [
                {"text": "I am happy", "emotions": [...]},
                {"text": "I am sad", "emotions": [...]}
            ]
        }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type doit être application/json"}), 400
    
    data = request.get_json()
    
    if 'texts' not in data or not isinstance(data['texts'], list):
        return jsonify({"error": "Le champ 'texts' doit être une liste"}), 400
    
    texts = data['texts']
    threshold = data.get('threshold', 0.5)
    
    try:
        results = []
        for text in texts:
            emotions_list, _ = predict_emotions(text, threshold=threshold)
            emotions_formatted = [
                {"emotion": emotion, "probability": round(prob, 2)}
                for emotion, prob in emotions_list
            ]
            results.append({
                "text": text,
                "emotions": emotions_formatted,
                "count": len(emotions_formatted)
            })
        
        return jsonify({
            "threshold": threshold,
            "results": results,
            "total": len(results)
        })
    
    except Exception as e:
        return jsonify({"error": f"Erreur: {str(e)}"}), 500


if __name__ == '__main__':
    print("="*60)
    print("API Emotion Detection démarrée")
    print("="*60)
    print("URL: http://localhost:5001")
    print("Documentation: http://localhost:5001/")
    print("\nExemple d'utilisation:")
    print("   curl -X POST http://localhost:5001/predict \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"text\": \"I am happy!\", \"threshold\": 0.5}'")
    print("="*60)
    
    # Lancer l'application
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
