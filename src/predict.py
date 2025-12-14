"""
Fonctions de prédiction d'émotions
===================================

Fonctions utilitaires pour prédire les émotions dans un texte.
"""

import tensorflow as tf
import os
from typing import Tuple, List, Optional

from .model import EmotionClassifier
from .config import LABEL_NAMES, DEFAULT_THRESHOLD


# Instance globale du modèle (chargée une seule fois)
_classifier_instance: Optional[EmotionClassifier] = None


def get_classifier() -> EmotionClassifier:
    """
    Retourne l'instance unique du classificateur (singleton pattern).
    
    Charge le modèle une seule fois et le réutilise ensuite.
    
    Returns:
        EmotionClassifier: Instance du modèle chargé.
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        print("Initialisation du modèle (première utilisation)...")
        _classifier_instance = EmotionClassifier()
        _classifier_instance.load_weights()
        print("Modèle prêt à l'emploi!")
    
    return _classifier_instance


def predict_emotions(
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
    return_all: bool = False
) -> Tuple[List[Tuple[str, float]], List[float]]:
    """
    Prédit les émotions présentes dans un texte.
    
    Cette fonction analyse un texte en anglais et retourne les émotions
    détectées avec leur probabilité (en pourcentage).
    
    Le modèle peut détecter plusieurs émotions simultanément (multi-label).
    
    Args:
        text: Texte à analyser (en anglais de préférence).
        threshold: Seuil de probabilité (0.0 à 1.0).
                   Les émotions au-dessus de ce seuil sont retournées.
                   Défaut: 0.50 (50%).
        return_all: Si True, retourne aussi toutes les probabilités brutes.
    
    Returns:
        Tuple contenant:
        - detected_emotions: Liste de tuples (émotion, probabilité%)
                            triée par probabilité décroissante.
        - all_probs: Liste de 28 probabilités (une par émotion) en %.
    
    Example:
        >>> emotions, probs = predict_emotions("I am so happy!", threshold=0.5)
        >>> print(emotions)
        [('joy', 91.2), ('excitement', 82.5)]
        
        >>> # Texte avec émotions multiples
        >>> emotions, _ = predict_emotions("I'm sad but also hopeful", threshold=0.4)
        >>> print(emotions)
        [('sadness', 87.3), ('optimism', 62.1)]
    
    Notes:
        - Le modèle ne détecte PAS l'intensité des émotions (a little, very, etc.)
        - Fonctionne mieux avec des textes courts à moyens (<500 mots)
        - Entraîné sur du texte anglais, peut avoir des résultats variables sur d'autres langues
    """
    # Charger le modèle (singleton, chargé une seule fois)
    classifier = get_classifier()
    
    # Prédiction (logits bruts)
    logits = classifier.predict([text], verbose=0)
    
    # Appliquer sigmoid pour obtenir les probabilités (0 à 1)
    probs = tf.nn.sigmoid(logits[0]).numpy()
    
    # Convertir en pourcentage
    probs_percent = probs * 100
    
    # Filtrer les émotions au-dessus du seuil
    detected_emotions = []
    for idx, prob in enumerate(probs):
        if prob >= threshold:
            detected_emotions.append((LABEL_NAMES[idx], probs_percent[idx]))
    
    # Trier par probabilité décroissante
    detected_emotions.sort(key=lambda x: x[1], reverse=True)
    
    if return_all:
        return detected_emotions, probs_percent.tolist()
    else:
        return detected_emotions, probs_percent.tolist()


def predict_emotions_batch(
    texts: List[str],
    threshold: float = DEFAULT_THRESHOLD
) -> List[List[Tuple[str, float]]]:
    """
    Prédit les émotions pour plusieurs textes en une seule fois (batch).
    
    Plus efficace que d'appeler predict_emotions() en boucle.
    
    Args:
        texts: Liste de textes à analyser.
        threshold: Seuil de probabilité pour la détection.
    
    Returns:
        Liste de listes de tuples (émotion, probabilité%) pour chaque texte.
    
    Example:
        >>> texts = ["I am happy", "I am sad", "I am angry"]
        >>> results = predict_emotions_batch(texts, threshold=0.5)
        >>> for i, emotions in enumerate(results):
        ...     print(f"{texts[i]}: {emotions}")
    """
    classifier = get_classifier()
    
    # Prédiction batch
    logits = classifier.predict(texts, verbose=0)
    probs = tf.nn.sigmoid(logits).numpy()
    probs_percent = probs * 100
    
    # Extraire les émotions pour chaque texte
    all_results = []
    for text_probs in probs:
        detected = []
        for idx, prob in enumerate(text_probs):
            if prob >= threshold:
                detected.append((LABEL_NAMES[idx], probs_percent[len(all_results)][idx]))
        detected.sort(key=lambda x: x[1], reverse=True)
        all_results.append(detected)
    
    return all_results


def get_emotion_names() -> List[str]:
    """
    Retourne la liste des 28 émotions détectables.
    
    Returns:
        Liste des noms d'émotions.
    """
    return LABEL_NAMES.copy()


def format_results(emotions: List[Tuple[str, float]], top_n: int = 5) -> str:
    """
    Formate joliment les résultats de prédiction.
    
    Args:
        emotions: Liste de tuples (émotion, probabilité%).
        top_n: Nombre d'émotions à afficher (défaut: 5).
    
    Returns:
        Chaîne formatée avec les émotions et leurs probabilités.
    
    Example:
        >>> emotions, _ = predict_emotions("I am happy!")
        >>> print(format_results(emotions))
        🎭 Émotions détectées:
          1. joy          : 91.2%
          2. excitement   : 82.5%
    """
    if not emotions:
        return "🎭 Aucune émotion détectée au-dessus du seuil."
    
    result = "🎭 Émotions détectées:\n"
    for i, (emotion, prob) in enumerate(emotions[:top_n], 1):
        result += f"  {i}. {emotion:15s} : {prob:.1f}%\n"
    
    return result.rstrip()
