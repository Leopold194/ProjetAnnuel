import projetannuel as pa 
from lib_gridfs import load_posters_with_size
import numpy as np 
import random
import matplotlib.pyplot as plt
import os 
import csv
import itertools
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

i=1
le = LabelEncoder()

### A plot: surface de la meilleure perf par rapport à la taille de posters (nombre de pixels) et le nombre de catégories (pour le fun)

### Paramètres globaux  
param_grid = {
    'categories': [["Horreur", "Animation"], ["Horreur", "Animation", "Action"]],
    'sizes': [(15,10)],
    'hidden_layers': [[8],[16],[32],[8,8],[16,16],[32,32],[8,16,8],[450],[450,450]],  # Nombre de neurones par couche cachée
    'learning_rate': [0.001, 0.01, 0.001],
    'epochs': [1000, 2000],
}

# dossiers
os.makedirs("results/logs", exist_ok=True)

#logs
log_path = "results/logs/mlp.csv"
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model_type",
            "categories",
            "image_size",
            "train_accuracy",
            "test_accuracy",
            "hidden_layers",
            "learning_rate",
            "epochs"
        ])


#### train
best_score = 0.0
best_params = None
last_size = None
last_categories=None

for size, categories, hidden_layers, lr, epochs in itertools.product(
    param_grid["sizes"],
    param_grid["categories"],
    param_grid["hidden_layers"],
    param_grid["learning_rate"],
    param_grid["epochs"],
):
    
    print(f"\n🧪 Testing size={size}, categories={len(categories)}, layers={hidden_layers}, lr={lr}, epochs={epochs}")

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

    # préparation des données
    X_train = imgs_as_lists_train
    X_test = imgs_as_lists_test
    labels = sorted(list(set(genres_train)))
    label_to_id = {label: i for i, label in enumerate(labels)}

    def one_hot(label):
        vec = [0.0] * len(labels)
        vec[label_to_id[label]] = 1.0
        return vec

    y_train = [one_hot(y) for y in genres_train]
    y_test = [one_hot(y) for y in genres_test]

    # architecture
    input_dim = len(X_train[0])
    output_dim = len(labels)
    layer_sizes = [input_dim] + hidden_layers + [output_dim]

    model = pa.MLP(layer_sizes, seed=42)

    # entraînement
    model.train(X_train, y_train, X_test, y_test, epochs, lr, True, 42)


    # Prédictions
    Y_pred_idx = [int(np.argmax(model.predict(x, True))) for x in X_test]

        # Mapping inverse des labels
    id_to_label = {i: label for label, i in label_to_id.items()}

    # 🔹 Prédictions TEST
    Y_pred_test_idx = [int(np.argmax(model.predict(x, True))) for x in X_test]
    Y_pred_test = [id_to_label[idx] for idx in Y_pred_test_idx]
    acc_test = pa.accuracy_score(genres_test, Y_pred_test)

    # 🔹 Prédictions TRAIN
    Y_pred_train_idx = [int(np.argmax(model.predict(x, True))) for x in X_train]
    Y_pred_train = [id_to_label[idx] for idx in Y_pred_train_idx]
    acc_train = pa.accuracy_score(genres_train, Y_pred_train)

    # ✅ Affichage
    print(f"✅ Accuracy Test  = {acc_test:.4f}")
    print(f"✅ Accuracy Train = {acc_train:.4f}")
    print(f"📌 Sample prediction: {Y_pred_test[:5]}")

    # 📁 Log CSV
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            "mlp",
            "|".join(categories),
            f"{size[0]}x{size[1]}",
            round(acc_train, 4),
            round(acc_test, 4),
            "-".join(map(str, hidden_layers)),
            lr,
            epochs
        ])
    
    os.makedirs("results/graphiques/mlp", exist_ok=True)

    archi = "-".join([str(x) for x in hidden_layers])
    # Generate filename based on parameters
    plot_path = f"results/graphiques/mlp/n_batch_loss_size{size[0]}x{size[1]}_cat{len(categories)}_lr{lr}_epochs{epochs}_layers{archi}.png"

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_loss, label='Training Loss', color='blue')
    plt.plot(model.test_loss, label='Test Loss', color='red')
    plt.title(f'Loss Curves\nSize: {size[0]}x{size[1]} | Categories: {len(categories)} | LR: {lr} | Epochs: {epochs}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"📊 Loss plot saved to {plot_path}")
            




