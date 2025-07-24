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
    'sizes' : [(15,10)],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "epochs": [1000, 5000],
    "algo":["rosenblatt","gradient-descent"]
}

# dossiers
os.makedirs("results/graphiques/linear", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

#logs
log_path = "results/logs/linear_scikit.csv"
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "model", "categories", "size", "train_accuracy", "test_accuracy", "epochs", "learning_rate", "algo"])


#### train

best_score = 0.0
best_params = None
last_size = None
last_categories=None

for size, categories, lr, epochs,algo in itertools.product(
    param_grid["sizes"],
    param_grid["categories"],
    param_grid["learning_rate"],
    param_grid["epochs"],
    param_grid["algo"]
):
    print(f"\n🧪 Testing size={size}, categories={len(categories)} {lr=}, {epochs=}, {algo=}")
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

    from sklearn.linear_model import Perceptron, SGDClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(genres_train)
    y_test = label_encoder.transform(genres_test)

    if algo == "rosenblatt":
        model = Perceptron(max_iter=epochs, eta0=lr, tol=None, shuffle=True, verbose=0)
    elif algo == "gradient-descent":
        model = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=lr, max_iter=epochs, tol=None, shuffle=True, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    model.fit(imgs_as_lists_train, y_train)

    # Train predictions
    Y_pred_train = model.predict(imgs_as_lists_train)
    acc_train = accuracy_score(y_train, Y_pred_train)

    # Test predictions
    Y_pred = model.predict(imgs_as_lists_test)
    acc = accuracy_score(y_test, Y_pred)
    
    #Log CSV
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            "linear",
            "|".join(categories),
            f"{size[0]}x{size[1]}",
            acc_train,
            acc,
            epochs,
            lr,
            algo            
        ])

    # MàJ meilleur modèle
    if acc > best_score:
        best_score = acc
        best_params = (size, categories,lr,epochs,algo)

"""
    y_train = pa.string_labels(genres_train)
    y_test = pa.string_labels(genres_test)
    
    model = pa.LinearModel(
        imgs_as_lists_train,
        y_train
    )
    
    model.train_classification(epochs=epochs, learning_rate=lr, algo=algo, x_test=imgs_as_lists_test, y_test=pa.string_labels(genres_test))

    #pred_test
    Y_pred = [model.predict(x) for x in imgs_as_lists_test]
    acc = pa.accuracy_score(genres_test, Y_pred)
    
    #pred_train
    Y_pred_train = [model.predict(x) for x in imgs_as_lists_train]
    acc_train = pa.accuracy_score(genres_train, Y_pred_train)



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
    graph_filename = f"results/graphiques/linear/size_{size[0]}x{size[1]}_lr{lr}_epochs{epochs}_{algo}.png"
    plt.savefig(graph_filename)
    plt.close()
"""
   