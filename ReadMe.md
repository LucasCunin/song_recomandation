````
.
├── ReadMe.md
├── data_class
│   ├── Audio.py
│   ├── AudioDataframe.py
│   └── feature_extractor.py.py
├── fleur.ipynb
├── models
│   ├── cluster.py
│   └── recommender.py
├── requirements.txt
├── run.py
└── tools
    └── preprocessing.py

3 directories, 10 files
````


## Organisation des fichiers du projet

### 1. **`audio.py`**  

* **Rôle** : Contient la classe principale `Audio`, qui prend en entrée un fichier audio ou un DataFrame audio.
* **Fonctionnalité** : Gère la récupération des features (caractéristiques audio) pour chaque extrait audio provenant de la base de données.
* **Explication** : Chaque objet `Audio` permet d’extraire des caractéristiques individuelles (zcr, rms, centroid, etc.) sur chaque extrait. 

### 2. **`feature_extractor.py`**

* **Rôle** : Fichier contenant la fonction principale d’itération sur l’ensemble de la base audio.
* **Fonctionnalité** : Cette fonction utilise la classe `Audio` pour extraire tous les features de chaque extrait présent dans la base, puis consolide les résultats.
* **Explication** : C’est ce script qui orchestre le traitement de masse.

### 3. **`distance.py`** (ou `cosine.py` ou `distance_metrics.py`)

* **Rôle** : Définir les fonctions de distance ou de similarité (par exemple, la distance cosinus) pour comparer les features extraits.
* **Fonctionnalité** : Fournit les méthodes pour calculer la distance ou la similarité entre deux ensembles de features audio. 

### 4. **`cluster.py`**

* **Rôle** : Fichier dédié à la mise en œuvre des algorithmes de clustering.
* **Fonctionnalité** : Permet de grouper les extraits audio similaires à partir des features extraits, en utilisant les distances définies précédemment.

### run.py 

pooint d'entré de l'app pour faire soit la recomandation avec la cosigne soit avec le clustering

### preprocessing.py

contient la ligc du preproc 


### **`data_class.py`** (optionnel, version V2)

* **Rôle** : Fournir une structure de données normalisée pour manipuler les DataFrames audio nettoyés.
* **Fonctionnalité** : Permet une manipulation plus robuste et typée des données audio, pour des traitements avancés ou des évolutions futures.




## Synthèse du workflow

* **`audio.py`** → Définit comment extraire les features d’un extrait audio.
* **`feature_extractor.py`** → Orchestration : parcourt la base, applique la classe Audio à chaque entrée, et centralise les features.
* **`distance.py` / `cosine.py`** → Fournit les métriques de comparaison.
* **`cluster.py`** → Regroupe les extraits en clusters sur la base de ces métriques.
* **`data_class.py`** → (facultatif, V2) Pour normaliser et fiabiliser la gestion des DataFrames.


