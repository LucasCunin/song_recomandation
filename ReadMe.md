# Système de Recommandation Audio

Ce projet implémente un système de recommandation audio basé sur la similarité des caractéristiques audio extraites. Il permet de trouver des morceaux de musique similaires à un morceau de référence en utilisant des caractéristiques audio comme les MFCC, le tempo, et d'autres descripteurs audio.

## Structure du Projet

```
.
├── data_class/
│   ├── __init__.py
│   ├── Audio.py              # Classes pour l'extraction des features audio
│   └── AudioDataframe.py     # DataFrame personnalisé pour les features audio
├── models/
│   └── recommender.py        # Système de recommandation par similarité
├── tools/
│   └── preprocessing.py      # Outils de prétraitement (scaling, PCA)
├── CHANGELOG.md             # Historique des changements et roadmap
├── VERSION                  # Version actuelle du projet
├── requirements.txt         # Dépendances du projet
└── run_recommendation.py    # Script principal de recommandation
```

## Installation

1. Cloner le repository
2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

### Recommandation Audio

Le système peut être utilisé en ligne de commande :

```bash
# Recommandation avec scaling des features
python run_recommendation.py --data_path Data/genres_original --query_path "chemin/vers/votre/audio.wav" --n_recommendations 5 --scale

# Recommandation sans scaling
python run_recommendation.py --data_path Data/genres_original --query_path "chemin/vers/votre/audio.wav" --n_recommendations 5
```

Options disponibles :

- `--data_path` : Chemin vers le dossier contenant les fichiers audio (défaut: "Data/genres_original")
- `--query_path` : Chemin vers le fichier audio de requête (obligatoire)
- `--n_recommendations` : Nombre de recommandations à retourner (défaut: 5)
- `--scale` : Activer le scaling des features (optionnel)
- `--force_extract` : Forcer la réextraction des features même si le cache existe

### Exemple d'Utilisation

```bash
# Exemple avec un fichier country
python run_recommendation.py --data_path Data/genres_original --query_path Data/genres_original/country/country.00002.wav --n_recommendations 5 --scale
```

Sortie attendue :

```
Chargement du cache : Data/genres_original/audio_features.csv
Application du scaling aux features...
Extraction des features du fichier de requête : Data/genres_original/country/country.00002.wav

Recommandations (chemins des fichiers) :
Data/genres_original/country/country.00031.wav
Data/genres_original/blues/blues.00000.wav
Data/genres_original/pop/pop.00063.wav
Data/genres_original/rock/rock.00000.wav
Data/genres_original/reggae/reggae.00084.wav
```

## Fonctionnalités

### Extraction de Features (`Audio.py`)
- Extraction des caractéristiques audio :
  - MFCC (13 coefficients avec moyenne et variance)
  - Caractéristiques temporelles (ZCR, RMS)
  - Caractéristiques spectrales (centroid, bandwidth, rolloff)
  - Tempo
  - F0 (fréquence fondamentale)
  - Contraste spectral

### Gestion des Données (`AudioDataframe.py`)
- DataFrame personnalisé héritant de pandas.DataFrame
- Chargement automatique depuis un dossier de fichiers audio
- Cache des features extraites dans un fichier CSV
- Support pour le chargement depuis un DataFrame existant

### Prétraitement (`preprocessing.py`)
- Scaling des features avec StandardScaler
- Réduction de dimensionnalité avec PCA (optionnel)
- Préservation des métadonnées lors des transformations

### Recommandation (`recommender.py`)
- Système de recommandation basé sur la similarité
- Support des features scaled et non-scaled
- Retourne les chemins des fichiers les plus similaires

## Workflow

1. **Chargement/Extraction des Features** :
   - Si un cache existe, les features sont chargées depuis le fichier CSV
   - Sinon, les fichiers audio sont traités et les features sont extraites
   - Les features sont stockées dans un AudioDataFrame

2. **Prétraitement** (si --scale est activé) :
   - Les features numériques sont standardisées
   - Les métadonnées (path, label) sont préservées

3. **Recommandation** :
   - Les features du fichier de requête sont extraites
   - Le système calcule la similarité avec tous les morceaux
   - Les n_recommendations morceaux les plus similaires sont retournés

## Notes

- Un avertissement peut apparaître concernant `librosa.beat.tempo` : c'est normal et n'affecte pas le fonctionnement
- Le système utilise un cache pour éviter de réextraire les features à chaque fois
- Le scaling des features peut améliorer la qualité des recommandations

## Gestion des Versions

Ce projet suit le [Semantic Versioning](https://semver.org/spec/v2.0.0.html) pour la numérotation des versions.

- Le fichier `VERSION` contient la version actuelle du projet
- Le fichier `CHANGELOG.md` documente :
  - Les changements pour chaque version
  - La roadmap des futures versions
  - Les bugs connus
  - Les suggestions d'amélioration

Pour voir les changements prévus et l'historique des versions, consultez le [CHANGELOG.md](CHANGELOG.md).