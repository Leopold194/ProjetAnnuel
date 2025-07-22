import projetannuel as pa 
from lib_gridfs import load_posters_with_size
import numpy as np 
import random
import matplotlib.pyplot as plt
import os 
import csv
import itertools
from datetime import datetime
### A plot: surface de la meilleure perf par rapport à la taille de posters (nombre de pixels) et le nombre de catégories (pour le fun)

### Paramètres globaux 
param_grid = {
    'categories' : [["Horreur",'Animation'],["Horreur","Animation","Action"]],
    'sizes' : [(10,10),(10,15),(15,15),(20,20),(15,20)],
    'C': [10, 100,500 ],
    'gamma': [1e-2, 1e-1, 1, 10, 100],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "epochs": [1000, 5000],
}

# dossiers
os.makedirs("results/graphiques", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

#logs
log_path = "results/logs/rbf.csv"
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "model", "categories", "size", "accuracy", "C", "gamma"])


#### train

best_score = 0.0
best_params = None
last_size = None
last_categories=None

for size, categories, c, gamma in itertools.product(
    param_grid["sizes"],
    param_grid["categories"],
    param_grid["C"],
    param_grid["gamma"],
):
    print(f"\n🧪 Testing size={size}, categories={len(categories)} C={c}, gamma={gamma}")
    if size != last_size or categories!=last_categories:
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
        
    last_size=size
    last_categories = categories    
    #modele

    y = pa.string_labels(genres_shuffled)
    model = pa.RBF(
        imgs_shuffled,
        y,
        gamma = gamma,
        k = c,
        seed = 42
    )
    model.train_classification(epochs=1_000, learning_rate=0.01, algo="gradient-descent", x_test=imgs_shuffled, y_test=pa.string_labels(genres_shuffled))

    Y_pred = [model.predict(x) for x in imgs_shuffled]
    acc = pa.accuracy_score(genres_shuffled, Y_pred)
    prop = Y_pred.count("Horreur") / len(Y_pred)

    print(f"✅ Accuracy = {acc:.4f} | Proportion de 'Horreur' : {prop:.2f}")

    #plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(model.train_loss, label='Train Loss')
    axs[0].plot(model.test_loss, label='Test Loss')
    axs[0].set_title("Courbe de perte")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(model.train_accuracy, label='Train Accuracy')
    axs[1].plot(model.test_accuracy, label='Test Accuracy')
    axs[1].set_title("Courbe d'accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    # Sauvegarde des courbes
    graph_filename = f"results/graphiques/size_{size[0]}x{size[1]}_C{c}_gamma{gamma}.png"
    plt.savefig(graph_filename)
    plt.close()

    #Log CSV
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            "rbf",
            "|".join(categories),
            f"{size[0]}x{size[1]}",
            round(acc, 4),
            c,
            gamma
        ])

    # MàJ meilleur modèle
    if acc > best_score:
        best_score = acc
        best_params = (gamma, c, size, categories)
