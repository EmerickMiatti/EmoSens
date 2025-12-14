# Notebooks - Expérimentation & Entraînement

Ce dossier contient tous les notebooks Jupyter pour l'expérimentation et l'entraînement des modèles.

## Contenu

### Notebooks d'entraînement
- **`BERT_Base_GoEmotions_COLAB.ipynb`** ⭐ **PRINCIPAL**
  - Entraînement du modèle BERT Base sur GoEmotions (43k exemples)
  - AUC: 0.85, Recall: 58%, Precision: 32%
  - Modèle sauvegardé dans `../models/bert_base_20251213_170236.weights.h5`
- **`finetuning_GoEmotions_tf.ipynb`** - Entrainement BERT Small

### Autres notebooks
- **`test_chargement_inference_bert_base.ipynb`** - Test du chargement du modèle BERT Base

## Utilisation

```bash
# Lancer Jupyter depuis la racine du projet
cd "c:\Users\Emerick\Documents\DL Project"
jupyter notebook notebooks/
```

## Résultats



Le meilleur modèle actuel :
- **Architecture** : BERT Base (110M paramètres)
- **Dataset** : GoEmotions (43k train, 5k validation)
- **Émotions** : 28 classes
- **Performance** : AUC 0.85
- **Poids sauvegardés** : `../models/bert_base_20251213_170236.weights.h5`

---

**Note : Le projet a initialement testé BERT Small, mais la version finale utilise BERT Base pour de meilleures performances. Certains anciens notebooks ou dossiers peuvent encore mentionner BERT Small, mais le modèle actif et la configuration sont bien sur BERT Base.**

## Notes

- Les notebooks contiennent tous les commentaires et markdown d'origine
- Pour l'utilisation en production, utilisez le module `src/` à la racine
- Les modèles entraînés sont sauvegardés dans `../models/`
