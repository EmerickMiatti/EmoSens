"""
EmoSense
==================================

Module pour la détection d'émotions multi-labels sur texte en anglais.
Utilise BERT Small fine-tuné sur le dataset GoEmotions (28 émotions).
"""

__version__ = "1.0.0"
__author__ = "Emerick MIATTI"

from .model import EmotionClassifier
from .predict import predict_emotions

__all__ = ['EmotionClassifier', 'predict_emotions']
