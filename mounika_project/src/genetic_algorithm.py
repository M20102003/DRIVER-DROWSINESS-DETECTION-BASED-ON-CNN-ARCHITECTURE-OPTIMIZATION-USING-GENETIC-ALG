import os
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Disable GPU before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf  # Import after setting the environment variable

# Load processed data
BASE_DIR = r"C:\Users\cnmou\Desktop\mounika_project\src"
PROCESSED_DIR =r"C:\Users\cnmou\Desktop\mounika_project\data\processed_data"

X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

# Add channel dimension
X_train = np.expand_dims(X_train, axis=-1)

# Genetic Algorithm Parameters
POPULATION_SIZE = 10
GENERATIONS = 5
MUTATION_RATE = 0.1

# Define CNN parameters
LAYERS = [1, 2, 3]
FILTERS = [32, 64, 128]
KERNEL_SIZES = [(3, 3), (5, 5)]
DENSE_UNITS = [128, 256]
DROPOUT_RATES = [0.3, 0.5]

# Fitness function
def fitness_function(individual):
    try:
        print(f"Evaluating individual: {individual}")
        model = create_model(individual)
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        accuracy = history.history["accuracy"][0]
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    except Exception as e:
        print(f"Error in fitness function: {e}")
        return 0

# Create CNN model
def create_model(individual):
    model = Sequential()
    model.add(Input(shape=(48, 48, 1)))
    num_layers = individual["num_layers"]
    
    for i in range(num_layers):
        filters = individual["filters"][i % len(individual["filters"])]
        kernel_size = individual["kernel_size"][i % len(individual["kernel_size"])]
        model.add(Conv2D(filters, kernel_size, activation="relu"))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(individual["dense_units"], activation="relu"))
    model.add(Dropout(individual["dropout_rate"]))
    model.add(Dense(1, activation="sigmoid"))
    return model

# Genetic Algorithm
def genetic_algorithm():
    # Initialize population
    population = []
    for _ in range(POPULATION_SIZE):
        num_layers = random.choice(LAYERS)
        individual = {
            "num_layers": num_layers,
            "filters": [random.choice(FILTERS) for _ in range(num_layers)],
            "kernel_size": [random.choice(KERNEL_SIZES) for _ in range(num_layers)],
            "dense_units": random.choice(DENSE_UNITS),
            "dropout_rate": random.choice(DROPOUT_RATES)
        }
        population.append(individual)

    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation + 1}/{GENERATIONS}")
        population = sorted(population, key=lambda x: fitness_function(x), reverse=True)
        best_individual = population[0]
        best_accuracy = fitness_function(best_individual)  # Avoid recomputing fitness
        print(f"Best Accuracy: {best_accuracy * 100:.2f}%")

        # Crossover and Mutation
        new_population = [best_individual]
        for i in range(POPULATION_SIZE - 1):
            parent1, parent2 = random.choices(population[:5], k=2)
            child = {
                "num_layers": random.choice([parent1["num_layers"], parent2["num_layers"]]),
                "filters": [random.choice([parent1["filters"][i], parent2["filters"][i]]) 
                            for i in range(min(len(parent1["filters"]), len(parent2["filters"])))],
                "kernel_size": [random.choice([parent1["kernel_size"][i], parent2["kernel_size"][i]]) 
                                for i in range(min(len(parent1["kernel_size"]), len(parent2["kernel_size"])))],
                "dense_units": random.choice([parent1["dense_units"], parent2["dense_units"]]),
                "dropout_rate": random.choice([parent1["dropout_rate"], parent2["dropout_rate"]])
            }
            if random.random() < MUTATION_RATE:
                child["num_layers"] = random.choice(LAYERS)
            new_population.append(child)

        population = new_population

    return best_individual

# Run Genetic Algorithm
if __name__ == "__main__":

    print("Starting Genetic Algorithm...")
    best_model_params = genetic_algorithm()
    print("\nGenetic Algorithm Completed!")
    print("Best Model Parameters:", best_model_params)

    # Ensure models directory exists
    import json
    best_architecture_file = os.path.join(BASE_DIR, "models", "best_architecture.json")
    os.makedirs(os.path.dirname(best_architecture_file), exist_ok=True)
    
    with open(best_architecture_file, "w") as f:
        json.dump(best_model_params, f)
    print(f"Best architecture saved to: {best_architecture_file}")