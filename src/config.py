
"""
Configuration du projet
========================

Chemins, paramètres et constantes pour le modèle d'émotions (BERT Base).

Note : Le preset utilisé est 'bert_base_en_uncased' (110M paramètres).
"""

import os

# === CHEMINS ===
# Dossier racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossier des modèles sauvegardés
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Chemin vers le dernier modèle entraîné
# Format TensorFlow natif (dossier avec .index et .data)
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bert_base_20251213_170236.weights.h5")

# === PARAMÈTRES DU MODÈLE ===
# Architecture BERT utilisée (BERT Base)
BERT_PRESET = "bert_base_en_uncased"

# Nombre de classes (émotions)
NUM_CLASSES = 28

# Longueur maximale des séquences (tokens)
SEQUENCE_LENGTH = 128

# === ÉMOTIONS (28 classes) ===
# Liste des 28 émotions de GoEmotions dans l'ordre exact de l'entraînement
LABEL_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# === PARAMÈTRES DE PRÉDICTION ===
# Seuil par défaut pour considérer une émotion comme détectée
DEFAULT_THRESHOLD = 0.50

# === GPU ===
# Activer la croissance dynamique de la mémoire GPU
GPU_MEMORY_GROWTH = True
