import projetannuel as pa
from lib_gridfs import load_posters_with_size
import numpy as np
import random
import os
import csv
import itertools
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import traceback

# Create result/logs directory
os.makedirs("results/logs", exist_ok=True)

# CSV log path
log_path = "results/logs/svm_linear.csv"
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "model_type", "categories", "image_size", "train_accuracy", "test_accuracy", "soft_margin"])

# Parameters
param_grid = {
    'categories': [["Horreur", "Animation", "Action"]],
    'sizes': [(15, 10)],
    "type": ["OneVsRest"],
    'C': [0.75,1.25,1.75,2.25,2.75,3,3.25,3.5,3.75,4.25,4.5],
}

param_combinations = list(itertools.product(
    param_grid["sizes"],
    param_grid["categories"],
    param_grid["C"],
    param_grid["type"]
))


# Train and evaluate one configuration
def train_and_evaluate(params):
    size, categories, c, classifier_type = params
    try:
        le = LabelEncoder()

        print(f"\n🧪 size={size}, categories={categories}, C={c}, classifier={classifier_type}")

        imgs, genres = load_posters_with_size(size, genres=categories)
        imgs_as_lists = np.array([img.tolist() for img in imgs]) / 255.0

        data = list(zip(imgs_as_lists, genres))
        random.shuffle(data)
        imgs_shuffled = [x[0] for x in data]
        genres_shuffled = [x[1] for x in data]
        genres_encoded = list(le.fit_transform(genres_shuffled))

        lim = int(len(data) * 0.8)
        X_train = imgs_shuffled[:lim]
        y_train = genres_encoded[:lim]
        X_test = imgs_shuffled[lim:]
        y_test = genres_encoded[lim:]

        kernel = pa.SVMKernelType.linear()
        if c=="hard":
            margin = pa.SoftMargin.Hard()
        else:
            margin = pa.SoftMargin.soft(c)

        if classifier_type == "OneVsRest":
            model = pa.SVMOvR(kernel, margin)
        else:
            model = pa.SVMOvO(kernel, margin)

        model.train(X_train, y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        acc_test = pa.accuracy_score(y_test, y_pred_test)
        acc_train = pa.accuracy_score(y_train, y_pred_train)

        print(f"✅ {classifier_type} | C={c} | test={acc_test:.4f} | train={acc_train:.4f}")

        return {
            "timestamp": datetime.now().isoformat(),
            "model_type": classifier_type,
            "categories": categories,
            "image_size": f"{size[0]}x{size[1]}",
            "train_accuracy": round(acc_train, 4),
            "test_accuracy": round(acc_test, 4),
            "soft_margin": c,
            "score": acc_test
        }

    except Exception as e:
        print("❌ Error in", params)
        traceback.print_exc()
        return None


if __name__ == "__main__":
    best_result = None
    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        for result in tqdm(executor.map(train_and_evaluate, param_combinations), total=len(param_combinations)):
            if result:
                results.append(result)
                if not best_result or result["score"] > best_result["score"]:
                    best_result = result

    # Write to log
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([
                r["timestamp"],
                r["model_type"],
                r["categories"],
                r["image_size"],
                r["train_accuracy"],
                r["test_accuracy"],
                r["soft_margin"]
            ])

    if best_result:
        print("\n🏆 Best configuration:")
        for k, v in best_result.items():
            print(f"{k}: {v}")
