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


le = LabelEncoder()

### A plot: surface de la meilleure perf par rapport à la taille de posters (nombre de pixels) et le nombre de catégories (pour le fun)

### Paramètres globaux 
param_grid = {
    'categories' : [["Horreur","Animation","Action"]],
    'sizes' : [(20,20)], #(10,15),
    'C': [0.1,0.5,1],
    'gamma': [1e-2, 5e-3, 5e-2],
    "type": ["OneVsRest"]#,"OneVsOne"]
}

# dossiers
os.makedirs("results/logs", exist_ok=True)

#logs
log_path = "results/logs/svm.csv"
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "model_type","categories", "image_size", "train_accuracy","test_accuracy", "Soft_Margin", "gamma","train_batch"])


#### train

best_score = 0.0
best_params = None
last_size = None
last_categories=None

for size, categories, c, gamma, classifier_type in itertools.product(
    param_grid["sizes"],
    param_grid["categories"],
    param_grid["C"],
    param_grid["gamma"],
    param_grid["type"]
):
    
    print(f"\n🧪 Testing size={size}, categories={len(categories)} C={c}, gamma={gamma}, classifier_type={classifier_type}")
    
    if size != last_size or categories!=last_categories:
        imgs, genres = load_posters_with_size(size, genres=categories)
        imgs_as_lists = np.array([img.tolist() for img in imgs]) / 255.0

        data = list(zip(imgs_as_lists, genres))
        random.shuffle(data)
        imgs_shuffled = [x[0] for x in data]
        genres_shuffled = [x[1] for x in data]
        genres_encoded = list(le.fit_transform(genres_shuffled))
        lim = int(len(data) * 0.8)
        imgs_as_lists_train = imgs_shuffled[:lim]
        genres_train = genres_encoded[:lim]
        imgs_as_lists_test = imgs_shuffled[lim:]
        genres_test = genres_encoded[lim:]
        
    last_size=size
    last_categories = categories    
    

    #modele
    kernel = pa.SVMKernelType.rbf(gamma)
    margin = pa.SoftMargin.soft(c)
    if classifier_type == "OneVsRest":
        model = pa.SVMOvR(kernel,margin)
    else:
        model = pa.SVMOvO(kernel,margin)
        
    model.train(imgs_as_lists_train,genres_train)

    #pred test
    Y_pred = model.predict(imgs_as_lists_test)
    acc = pa.accuracy_score(genres_test, Y_pred)
    
    #pred_train
    Y_pred_train = model.predict(imgs_as_lists_train)
    acc_train = pa.accuracy_score(genres_train, Y_pred_train)
    
    print(f"✅ Accuracy = {acc:.4f} | Accuracy_Train: = {acc_train:.4f}")


    #Log CSV
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            classifier_type,
            categories,
            f"{size[0]}x{size[1]}",
            round(acc_train, 4),
            round(acc, 4),
            c,
            gamma
        ])



    # MàJ meilleur modèle
    if acc > best_score:
        best_score = acc
        best_params = (gamma, c, size, categories)
