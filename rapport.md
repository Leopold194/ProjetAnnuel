# Rapport ProjetAnnuel

## Sommaire

1. Présentation du projet 
2. Présentation méthode
3. Dataset
3. Création modèles
4. Entrainement des modèles
5. Résultats
6. Discussion


## Présentation du projet

Le projet consiste à créer un classificateur de genre pour les affiches de film de cinéma.
L'utilité est surtout de développer des modèles, et de tester différentes architectures et approches afin de réaliser une application fonctionnelle et précise.

## Présentation méthode

3 grandes parties (1 et 2 sont indépendantes)

1. Constitution du dataset
2. Création de la librairie de ML
3. Entraînement de modèles 


## Dataset

### Sources de données

La création du dataset se fait directement via l'api TMDB, qui recense la plupart des films des dernières décennies.
Il s'agit de donnée publiquement accessible et facile à récupérer.

### Présentation du dataset

Le dataset est composé d'images labellisées par catégorie de films:
- Il y a 17 catégories de films
- pour 17k images en tout

Les catégories sont bien harmonisées, avec ~1k enregistrements par catégorie 

On redéfinit les tailles des images à 256x256 pour des raisons de mémoire (on entrainera jamais les images 3000*1500) et de normalisation

### Stockage dataset
Sur mongodb avec notre lib gridfs

## Création des modèles + lib

Actuellement la liste des modèles présents sur la lib pa est la suivante:

- Régression linéaire
- Classificateur linéaire
- PMC/MLP
- classificateur RBF
- SVM

Ces modèles implémentent tous les trait sérialisable pour pouvoir les utiliser directement dans le PA

La lib comprend aussi des fonctions comme par exemple:
accuracy_score()
...

### Régression linéaire

- new: crée un modèle MLP:

- Train
- Predict
- getter sur les weights

### Classifieur linéaire


### PMC / MLP


### RBF


### SVM



### Entrainement de modèles

Ici, on utilise des tailles d'images réduites.

En effet, on a ~1k entrées par classe, on ne peut donc pas garder des images 256x256 (~200k features) car sinon les modèles ne généralisent pas.

On va donc faire un resize simple et ramener les images à différentes tailles avec:
- comme taille 10x10 (300 features), 15x10 (450 features et gardent le ratio), 20x20 (1200 features)

On cherche à optimiser les grid de paramètres avec différentes approches:
- Estimation des paramètres en amont
    - RBF/SVM: + algorithme pour aller plus simple