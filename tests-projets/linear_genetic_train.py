import random
import projetannuel as pa 
from lib_gridfs import load_posters_with_size
import numpy as np 
import random
import matplotlib.pyplot as plt
import os 
import csv
import itertools
from datetime import datetime

def genetic_tune_continuous(train_model,
                            discrete_space,
                            continuous_space,
                            population_size=15,
                            generations=40,
                            crossover_rate=0.7,
                            mutation_rate=0.1,
                            mutation_scale=0.1):
    """
    Genetic tuning with support for discrete and continuous hyperparameters.

    Args:
        train_model: function accepting **params dict and returning test accuracy.
        discrete_space: dict {param_name: list_of_values}
        continuous_space: dict {param_name: (min_value, max_value)}
        population_size: number of individuals
        generations: number of generations
        crossover_rate: probability of crossover
        mutation_rate: probability of mutation per gene
        mutation_scale: relative scale of gaussian mutation for continuous params

    Returns:
        best_params: dict of best hyperparameters
        best_acc: best accuracy found
    """
    # Combine both spaces for initialization
    def sample_individual():
        indiv = {}
        for k, vals in discrete_space.items():
            indiv[k] = random.choice(vals)
        for k, (low, high) in continuous_space.items():
            indiv[k] = random.uniform(low, high)
        return indiv

    # Initialize population
    population = [sample_individual() for _ in range(population_size)]
    best_params = None
    best_acc = -float('inf')

    for gen in range(1, generations + 1):
        # Evaluate fitness
        fitnesses = []
        for indiv in population:
            acc = train_model(**indiv)
            fitnesses.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_params = indiv.copy()

        # Selection: roulette wheel
        min_fit = min(fitnesses)
        shifted = [f - min_fit + 1e-6 for f in fitnesses]
        total = sum(shifted)
        probs = [f / total for f in shifted]

        new_pop = []
        while len(new_pop) < population_size:
            # Parent selection
            p1, p2 = random.choices(population, weights=probs, k=2)

            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = {}, {}
                # Discrete genes: uniform crossover
                for k in discrete_space:
                    if random.random() < 0.5:
                        c1[k], c2[k] = p1[k], p2[k]
                    else:
                        c1[k], c2[k] = p2[k], p1[k]
                # Continuous genes: arithmetic crossover
                for k in continuous_space:
                    alpha = random.random()
                    c1[k] = alpha * p1[k] + (1 - alpha) * p2[k]
                    c2[k] = alpha * p2[k] + (1 - alpha) * p1[k]
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            for child in (c1, c2):
                # Discrete mutation
                for k, vals in discrete_space.items():
                    if random.random() < mutation_rate:
                        child[k] = random.choice(vals)
                # Continuous mutation
                for k, (low, high) in continuous_space.items():
                    if random.random() < mutation_rate:
                        perturb = random.gauss(0, mutation_scale * (high - low))
                        child[k] = min(max(child[k] + perturb, low), high)
                new_pop.append(child)
                if len(new_pop) >= population_size:
                    break

        population = new_pop
        print(f"Gen {gen}/{generations} — Best so far: acc={best_acc:.4f}, params={best_params}")

    return best_params, best_acc


os.makedirs("results/graphiques/linear_genetic", exist_ok=True)


#logs
log_path = "results/logs/linear_genetic.csv"
if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "model", "categories", "size", "train_accuracy", "test_accuracy", "epochs", "learning_rate", "algo"])


imgs, genres = load_posters_with_size((15,10), genres=["Horreur","Animation","Action"])
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
y_train = pa.string_labels(genres_train)
y_test = pa.string_labels(genres_test)


def train_model(epochs, learning_rate, algo):
    epochs = int(epochs)
    model = pa.LinearModel(
        imgs_as_lists_train,
        y_train
    )
    

    model.train_classification(epochs=int(epochs), learning_rate=learning_rate, algo=str(algo), x_test=imgs_as_lists_test, y_test=pa.string_labels(genres_test))

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
    graph_filename = f"results/graphiques/linear_genetic/size_{15}x{10}_lr{learning_rate}_epochs{epochs}_{algo}.png"
    plt.savefig(graph_filename)
    plt.close()
    
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            "linear",
            str(["Horreur","Animation","Action"]),
            f"15x10",
            round(acc_train,4),
            round(acc,4),
            epochs,
            round(learning_rate,4),
            algo            
        ])
    
    return acc

# Define your search spaces:
discrete_space = {
    "algo": ["gradient-descent"]
}

continuous_space = {
    "learning_rate": (0.0001, 0.1),
    "epochs": (500, 1500)
}

genetic_tune_continuous(train_model, discrete_space, continuous_space)

