import projetannuel as pa 
from lib_gridfs import load_posters_with_size
import numpy as np 
import random
import time
import matplotlib.pyplot as plt
import os 
import csv
import itertools
from datetime import datetime
### A plot: surface de la meilleure perf par rapport à la taille de posters (nombre de pixels) et le nombre de catégories (pour le fun)

### Paramètres globaux 
# param_grid = {
#     'categories' : [["Horreur","Animation","Action"]],
#     'sizes' : [(15,10)],
#     'C': [10, 100, 500],
#     'gamma': [1e-2, 1e-1, 1],
#     "learning_rate": [0.001, 0.01, 0.05, 0.1],
#     "epochs": [1000, 5000],
#     "algo": ["rosenblatt", "gradient-descent"]
# }
g = [{
      "id": 28,
      "name": "Action"
    },
    {
      "id": 12,
      "name": "Aventure"
    },
    {
      "id": 16,
      "name": "Animation"
    },
    {
      "id": 35,
      "name": "Comédie"
    },
    {
      "id": 80,
      "name": "Crime"
    },
    {
      "id": 99,
      "name": "Documentaire"
    },
    {
      "id": 18,
      "name": "Drame"
    },
    {
      "id": 10751,
      "name": "Familial"
    },
    {
      "id": 14,
      "name": "Fantastique"
    },
    {
      "id": 36,
      "name": "Histoire"
    }]

genres_full = [m["name"] for m in g]

param_grid = {
    'categories' : [["Horreur","Animation"]],
    'sizes' : [(15,10), (10, 10), (20, 15), (10, 5)],
    'C': [500],
    'gamma': [1e-1],
    "learning_rate": [0.05],
    "epochs": [1000],
    "algo": ["gradient-descent"]
}

# 2025-07-22T12:42:00.380519,rbf,Horreur|Animation,15x10,0.795,0.78,500,0.1,gradient-descent,5000,0.05


# dossiers
os.makedirs("results/graphiques", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

#logs
log_path = "results/logs/rbf_various_sizes.csv"
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "model", "categories", "size", 
            "train_accuracy", "test_accuracy", 
            "C", "gamma", "algo", "epochs", "learning_rate",
            "train_duration_seconds"
        ])

#### train

best_score = 0.0
best_params = None
last_size = None
last_categories=None

for size, categories, c, gamma, algo, epochs, learning_rate in itertools.product(
    param_grid["sizes"],
    param_grid["categories"],
    param_grid["C"],
    param_grid["gamma"],
    param_grid["algo"],
    param_grid["epochs"],
    param_grid["learning_rate"]
):
    print(f"\n🧪 Testing size={size}, categories={len(categories)} C={c}, gamma={gamma}, algo={algo}, epochs={epochs}, learning_rate={learning_rate}")
    if size != last_size or categories != last_categories:
        imgs, genres = load_posters_with_size(size, genres=categories)
        imgs_as_lists = np.array([img.tolist() for img in imgs]) / 255.0

        data = list(zip(imgs_as_lists, genres))
        random.shuffle(data)
        imgs_shuffled = [x[0] for x in data]
        genres_shuffled = [x[1] for x in data]

        lim = int(len(data) * 0.8)
        imgs_as_lists_train = imgs_shuffled[:lim]
        genres_train = genres_shuffled[:lim]
        imgs_as_lists_test = imgs_shuffled[lim:]
        genres_test = genres_shuffled[lim:]
        
    last_size = size
    last_categories = categories    

    y = pa.string_labels(genres_train)
    model = pa.RBF(
        imgs_as_lists_train,
        y,
        gamma=gamma,
        k=c,
        seed=42
    )

    # Mesure du temps d'entrainement
    start_time = time.time()
    model.train_classification(
        epochs=epochs,
        learning_rate=learning_rate,
        algo=algo,
        x_test=imgs_as_lists_test,
        y_test=pa.string_labels(genres_test)
    )
    train_duration = time.time() - start_time

    Y_pred_train = [model.predict(x) for x in imgs_as_lists_train]
    acc_train = pa.accuracy_score(genres_train, Y_pred_train)
    
    Y_pred_test = [model.predict(x) for x in imgs_as_lists_test]
    acc_test = pa.accuracy_score(genres_test, Y_pred_test)
    
    prop = Y_pred_test.count("Horreur") / len(Y_pred_test)

    print(f"✅ Accuracy Train = {acc_train:.4f}")
    print(f"✅ Accuracy Test = {acc_test:.4f} | Proportion de 'Horreur' : {prop:.2f}")
    print(f"⏱️ Temps d'entraînement : {train_duration:.2f} secondes")

    # Sauvegarde dans le CSV avec le temps d'entraînement en plus
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            "rbf",
            "|".join(categories),
            f"{size[0]}x{size[1]}",
            round(acc_train, 4),
            round(acc_test, 4),
            c,
            gamma,
            algo,
            epochs,
            learning_rate,
            round(train_duration, 2)
        ])

    # MàJ meilleur modèle
    if acc_test > best_score:
        best_score = acc_test
        best_params = (gamma, c, size, categories, algo, epochs, learning_rate)
