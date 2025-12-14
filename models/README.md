
# Modèles - Poids sauvegardés (BERT Base)

Ce dossier contient tous les modèles entraînés.

## Contenu

### Modèles principaux
- **`bert_base_20251213_170236.weights.h5`**
  - BERT Base fine-tuné sur GoEmotions
  - 43k exemples d'entraînement
  - AUC: 0.85, Recall: 58%, Precision: 32%
  - Taille: ~420 MB
- **`bert_base_20251213_002923.weights.h5`**
  - Version précédente, même format


- Les modèles principaux sont au format `.weights.h5`
- Pour charger un autre modèle, modifiez `DEFAULT_MODEL_PATH` dans `src/config.py`

## Utilisation




Le module `src/` charge automatiquement le modèle actif depuis ce dossier (BERT Base) :

```python
from src import predict_emotions

# Le modèle est chargé depuis models/bert_base_20251213_170236.weights.h5
emotions, probs = predict_emotions("I am happy!")
```

## Configuration


Le chemin et le preset sont définis dans [`src/config.py`](../src/config.py) :
```python
BERT_PRESET = "bert_base_en_uncased"
DEFAULT_MODEL_PATH = "models/bert_base_20251213_170236.weights.h5"
```

## Notes

- Les modèles principaux sont au format `.weights.h5` (Keras/TensorFlow)
- Un export SavedModel complet est disponible dans `notebooks/bert_base_savedmodel_20251213_000915/`
- Pour charger un autre modèle, modifiez `DEFAULT_MODEL_PATH` dans `src/config.py`
