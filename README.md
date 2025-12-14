
disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, optimism, pride, realization,
relief, remorse, sadness, surprise, neutral

# Détection d'émotions avec BERT Base (GoEmotions)

Ce projet permet de détecter plusieurs émotions dans des textes anglais grâce à un modèle BERT Base fine-tuné sur le dataset GoEmotions (28 émotions).

## Performances

- AUC : 0.85
- Rappel : 58%
- Précision : 32%
- Dataset : ~43 000 exemples GoEmotions

## Installation

```bash
git clone <repo>
cd <repo>
pip install -r requirements_model.txt
```

## Utilisation rapide

```python
from src.predict import predict_emotions
emotions, probs = predict_emotions("I am so happy and excited!", threshold=0.5)
for emotion, prob in emotions:
    print(f"{emotion}: {prob:.1f}%")
```

## Structure du projet

```
notebooks/      # Notebooks d'entraînement et d'expérimentation
models/         # Modèles sauvegardés (.weights.h5)
src/            # Code source Python (production)
app.py          # API Flask pour déploiement
requirements_model.txt
README.md
```


## API Flask

Lancez l'API avec :

```bash
python app.py
```

Puis faites une requête POST sur `/predict` avec un texte à analyser.

## Limitations

- Entraîné uniquement sur l'anglais

## API avancée

### Prédiction batch (plusieurs textes)

```python
from src.predict import predict_emotions_batch

texts = [
    "I am happy",
    "I am sad and disappointed",
    "This is annoying!"
]

results = predict_emotions_batch(texts, threshold=0.50)

for text, emotions in zip(texts, results):
    print(f"{text}: {emotions}")
```

### Ajuster le seuil

```python
# Seuil bas (40%) = plus d'émotions détectées
emotions, _ = predict_emotions(text, threshold=0.40)

# Seuil haut (60%) = seulement émotions très fortes
emotions, _ = predict_emotions(text, threshold=0.60)
```

## Configuration

Le projet utilise **BERT Base** (`bert_base_en_uncased`).
Pour modifier la configuration, éditez `src/config.py` :

```python
# Chemin du modèle
DEFAULT_MODEL_PATH = "models/bert_base_20251213_170236.weights.h5"

# Preset BERT utilisé
BERT_PRESET = "bert_base_en_uncased"

# Seuil par défaut
DEFAULT_THRESHOLD = 0.50

# Activer GPU memory growth
GPU_MEMORY_GROWTH = True
```


## Tests

```bash
# Exemple de test simple
python -c "from src.predict import predict_emotions; \
           emotions, _ = predict_emotions('I am happy'); \
           print(emotions)"
```

## Ré-entraînement

Pour ré-entraîner le modèle sur vos propres données:

1. Ouvrir `finetuning_GoEmotions_tf.ipynb`
2. Modifier le dataset dans la cellule 5
3. Exécuter toutes les cellules
4. Le nouveau modèle sera sauvegardé dans `models/` ou dans un dossier horodaté selon le script utilisé

### Hyperparamètres d'entraînement:



## 🎓 Exemples d'utilisation

### Analyse de sentiment client

```python
from src.predict import predict_emotions

review = "The product is amazing! Fast delivery and great customer service."
emotions, _ = predict_emotions(review, threshold=0.40)

# Résultat: admiration, approval, gratitude
```

### Détection d'émotions négatives

```python
text = "This is the worst experience ever. I'm so disappointed."
emotions, _ = predict_emotions(text, threshold=0.50)

# Résultat: disappointment, anger, annoyance
```

### Analyse de texte long

```python
long_text = """
I just found out I got the job! I'm so excited and grateful.
But I'm also nervous about leaving my current team...
"""
emotions, _ = predict_emotions(long_text, threshold=0.40)

# Résultat: joy, excitement, gratitude, nervousness
```

---
