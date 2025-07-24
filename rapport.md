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

accessibilité via lib (blazingly fast via optimisations intelligentes )

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



### Train MLP

Ajout de la train + test loss -> Temps d'entrainement beaucoup plus long
-> En effet, à chaque epoch on doit calculer la loss sur tout le dataset, ce qui ralentit beaucoup l'entrainement:

#### Calcul rapide

**Configuration :**
- Exemples totaux : 3000  
  - Train (80%) : 2400  
  - Test (20%) : 600  
- Architecture :  
  - Entrées d = 450  
  - Couche cachée h = 8  
  - Sorties o = 3  


1. Paramètres du modèle

| Connexion        | Nombre de poids           | Calcul               |
|------------------|---------------------------|----------------------|
| Input → Hidden   | (d + 1) × h               | (450 + 1) × 8 = 3608 |
| Hidden → Output  | (h + 1) × o               | (8 + 1) × 3 = 27     |
| **Total W**      |                           | 3608 + 27 = 3635     |


2. **Coût forward & backprop (SGD) par epoch**

2.1 Coût par exemple  
- **Forward pass**  
  - d × h = 450 × 8 = 3600 ops  
  - h × o = 8 × 3 = 24 ops  
  - **Total forward = 3600 + 24 = 3624 ops**  
- **Backward pass** = 3624 ops  
- **Mise à jour des poids** = W = 3635 ops  
- **Total par exemple = 3624 + 3624 + 3635 = 10883 ops**

2.2 Coût total (2400 itérations SGD)  
- 2400 × 10883 ≈ 26 119 200 ops

---

3. **Coût calcul des pertes (train_loss + test_loss) par epoch**

3.1 Coût forward + loss par exemple  
- Forward pass = 3624 ops  
- Cross‐entropy (~10 ops/output × 3 outputs = 30 ops)  
- **Total par exemple = 3624 + 30 = 3654 ops**

3.2 Train_loss (2400 exemples)  
- 2400 × 3654 = 8 769 600 ops

3.3 Test_loss (600 exemples)  
- 600 × 3654 = 2 192 400 ops

3.4 Coût global calcul des pertes  
- 8 769 600 + 2 192 400 = 10 962 000 ops

---

4. **Récapitulatif des coûts par epoch**

| Étape                                | Ops par epoch     |
|--------------------------------------|-------------------|
| Propagation + update (SGD)           | ~ 26 119 200 ops  |
| Calcul global des pertes (train+test)| ~ 10 962 000 ops  |
| **Total**                            | **~ 37 081 200 ops** |

Entre 25 et 30% de temps de train supplémentaire.

Premier batch d'entrainement:

```python 
param_grid = {
    'categories': [["Horreur", "Animation"], ["Horreur", "Animation", "Action"]],
    'sizes': [(15,10)],
    'hidden_layers': [[8],[16],[32],[8,8],[16,16],[32,32],[8,16,8],[450],[450,450]],  # Nombre de neurones par couche cachée
    'learning_rate': [0.001, 0.01],
    'epochs': [1000, 2000],
}
```

Il semble que plus on ajoute de couche au modèle, plus le modèle devient mauvais en terme de prédiction.

Plus on ajoute de neurones aux couches aussi, apparemment les modèles des couches plus petites.

Sur le nombre d'epochs, on semble atteindre la convergence avec 2000 epochs dans la plupart des cas, meme si on peut augmenter encore un peu.

Quant au learning rate, il semble bien, meme si à 0.01 on commence à avoir des petits tressaillements, c'est surement la limite supérieure acceptable

De plus, le MLP, pour ces tailles est très peu consommateur en mémoire, on peut donc paralléliser l'entrainement de modèles sur Python.

On a aussi testé un modèle avec plus de paramètres et les résultats sont pas bons: 
- La loss augmente avec les epochs T-T
cf 1 couche à 450 neurones 

Batch 2:

Un algorithme génétique semble pertinent, on va le mettre en place pour voir ce que ca va donner.
(temps d'entrainement à voir).

L'objectif de ce batch est d'entrainer plus de modèles pour essayer d'améliorer la précision 



### SVM Linéaire:

Parallel/Concurrent training pour exploiter le CPU - Actuellement, le code est monothread et les CPU actuels en ont ~12 donc on met en place ca pour gagner du temps.

Affinage hyperparamètre en supposant qu'il y a un maximum local.

Le premier batch avait un C à 0.5 en meilleur paramètre, on fait donc un entrainement autour de cette valeur (avec les autres valeurs qui rentrent en compte)


#### Batch 2: augmentation quantité hyperparamètres
Résultats du 2eme batch:
2025-07-23T23:32:07.537398,OneVsRest,"['Horreur', 'Animation', 'Action']",15x10,0.4176,0.4271,0.2
2025-07-23T23:31:24.883152,OneVsRest,"['Horreur', 'Animation', 'Action']",15x10,0.4558,0.4271,0.3
2025-07-23T23:32:35.748839,OneVsRest,"['Horreur', 'Animation', 'Action']",15x10,0.5463,0.5193,0.4
2025-07-23T23:33:06.807763,OneVsRest,"['Horreur', 'Animation', 'Action']",15x10,0.4432,0.4322,0.6




#### Test 3: HardMargin
On se rend compte que ce n'est pas linéaire, on teste aussi le modèle sans le paramètre C:
Résultat du modèle:




### Test 4: 3eme batch de C, on essaie avec des C plus grands.
Pour des soucis de vitesse d'exécution, nous avons mis les paramètres 