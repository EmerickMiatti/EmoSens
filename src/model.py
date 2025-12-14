
"""
Chargement et gestion du modèle BERT
=====================================

Classe pour charger le modèle BERT Base fine-tuné sur GoEmotions.
"""

import tensorflow as tf
import keras
import keras_nlp
from typing import Optional
import os

from .config import (
    BERT_PRESET,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    DEFAULT_MODEL_PATH,
    GPU_MEMORY_GROWTH
)


class EmotionClassifier:
    """
    Classificateur d'émotions basé sur BERT Base.
    
    Détecte 28 émotions dans un texte (multi-label classification).
    Basé sur le dataset GoEmotions de Google.
    
    Architecture:
        - BERT Base (110M paramètres)
        - Sequence length: 128 tokens
        - 28 classes (émotions)
    
    Example:
        >>> model = EmotionClassifier()
        >>> model.load_weights("path/to/weights")
        >>> logits = model.predict("I am very happy!")
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le classificateur d'émotions.
        
        Args:
            model_path: Chemin vers les poids du modèle.
                       Si None, utilise DEFAULT_MODEL_PATH.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.classifier = None
        self._configure_gpu()
        self._build_model()
    
    def _configure_gpu(self):
        """Configure la mémoire GPU pour éviter les erreurs OOM."""
        if not GPU_MEMORY_GROWTH:
            return
            
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU détecté: {len(gpus)} GPU(s) configuré(s)")
            except RuntimeError as e:
                print(f"Erreur config GPU: {e}")
    
    def _build_model(self):
        """
        Construit l'architecture du modèle BERT.
        
        Crée un BertClassifier à partir du preset 'bert_base_en_uncased'
        avec 28 classes de sortie (une par émotion).
        """
        print("Construction du modèle BERT...")
        
        self.classifier = keras_nlp.models.BertClassifier.from_preset(
            BERT_PRESET,
            num_classes=NUM_CLASSES,
        )
        
        self.classifier.preprocessor.sequence_length = SEQUENCE_LENGTH
        
        print(f"Modèle construit: {BERT_PRESET}")
        print(f"Paramètres: 110M (BERT Base)")
        print(f"Classes: {NUM_CLASSES} émotions")
        print(f"Séquence max: {SEQUENCE_LENGTH} tokens")
    
    def load_weights(self, weights_path: Optional[str] = None):
        """
        Charge les poids entraînés du modèle.
        
        Args:
            weights_path: Chemin vers les poids. Si None, utilise self.model_path.
        
        Raises:
            FileNotFoundError: Si le chemin n'existe pas.
        """
        path = weights_path or self.model_path
        
        # Gestion des formats: .h5 (fichier unique) ou TensorFlow (dossier avec .index)
        if path.endswith('.h5') or path.endswith('.weights.h5'):
            # Format H5 direct
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Poids du modèle introuvables: {path}\n"
                    f"Assurez-vous d'avoir entraîné et sauvegardé le modèle."
                )
        else:
            # Format TensorFlow (avec .index)
            index_file = path + ".index"
            if not os.path.exists(index_file):
                raise FileNotFoundError(
                    f"Poids du modèle introuvables: {path}\n"
                    f"Fichier attendu: {index_file}\n"
                    f"Assurez-vous d'avoir entraîné et sauvegardé le modèle."
                )
        
        print(f"Chargement des poids depuis: {path}")
        self.classifier.load_weights(path)
        print("Poids chargés avec succès!")
    
    def predict(self, texts: list[str], verbose: int = 0):
        """
        Prédit les émotions pour une liste de textes.
        
        Args:
            texts: Liste de textes à analyser (en anglais).
            verbose: Niveau de verbosité (0 = silencieux, 1 = barre de progression).
        
        Returns:
            Logits bruts (avant activation sigmoid) de shape (batch_size, 28).
            Utilisez tf.nn.sigmoid() pour obtenir les probabilités.
        
        Exemple:
            >>> logits = model.predict(["I am happy", "I am sad"])
            >>> probs = tf.nn.sigmoid(logits).numpy()
        """
        if self.classifier is None:
            raise RuntimeError("Modèle non initialisé. Appelez load_weights() d'abord.")
        
        return self.classifier.predict(texts, verbose=verbose)
    
    def get_model(self):
        """Retourne le modèle Keras sous-jacent."""
        return self.classifier
