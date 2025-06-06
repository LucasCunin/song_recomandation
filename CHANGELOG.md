# Changelog


Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère à [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [TODO] 0.1.0 - 2025-06-XX

### Optimisé
- Refactorisation complète du code pour une meilleure lisibilité
- Optimisation des performances de lecture de la base de données
- Amélioration de la gestion de la mémoire
- Nettoyage des imports et de la structure du code
- Documentation améliorée du code

### Modifié
- Restructuration des classes pour une meilleure modularité
- Amélioration de la gestion des erreurs
- Optimisation de l'extraction des features audio
- Meilleure gestion du cache des features

### Corrigé
- Correction des fuites de mémoire dans l'extraction audio
- Amélioration de la gestion des fichiers temporaires
- Correction des problèmes de performance dans le chargement des données

## [0.0.2] - 2025-06-XX

### Ajouté
- Système de clustering pour regrouper les musiques similaires
- Visualisation des clusters de musiques
- Métriques d'évaluation des clusters
- Support de différents algorithmes de clustering (K-means, DBSCAN)
- Interface pour explorer les clusters

### Modifié
- Adaptation du système de recommandation pour utiliser les clusters
- Amélioration de la similarité en prenant en compte les clusters
- Modification de la structure de données pour supporter le clustering

## [0.0.1] - 2025-06-05

### Ajouté
- Système de base de recommandation audio
- Extraction des features audio (MFCC, ZCR, RMS, etc.)
- Support du scaling des features
- Système de cache pour les features extraites
- Interface en ligne de commande
- Documentation initiale
