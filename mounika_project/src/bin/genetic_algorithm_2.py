# import os
# import random



# # Install missing libraries
# try:
#     import numpy as np
# except ImportError:
#     os.system("pip install numpy")
#     import numpy as np

# try:
#     import tensorflow as tf
# except ImportError:
#     os.system("pip install tensorflow")
#     import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# # Set base directory dynamically
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_data")

# # Load processed data
# X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
# y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

# # Add channel dimension
# X_train = np.expand_dims(X_train, axis=-1)

# # Genetic Algorithm Parameters
# POPULATION_SIZE = 10
# GENERATIONS = 5
# MUTATION_RATE = 0.1

# # Define CNN parameters to optimize
# LAYERS = [1, 2, 3]
# FILTERS = [32, 64, 128]
# KERNEL_SIZES = [(3, 3), (5, 5)]
# DENSE_UNITS = [128, 256]
# DROPOUT_RATES = [0.3, 0.5]

# # Fitness function
# def fitness_function(individual):
#     try:
#         model = create_model(individual)
#         model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
#         history = model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)
#         return history.history["accuracy"][0]
#     except Exception as e:
#         print(f"Error in fitness function: {e}")
#         return 0  # Return 0 fitness for invalid individuals

# # Create CNN model from individual
# def create_model(individual):
#     model = Sequential()
#     num_layers = individual["num_layers"]
#     for i in range(num_layers):
#         # Ensure filters and kernel_size have enough values
#         filters = individual["filters"][i] if i < len(individual["filters"]) else individual["filters"][-1]
#         kernel_size = individual["kernel_size"][i] if i < len(individual["kernel_size"]) else individual["kernel_size"][-1]
#         model.add(Conv2D(filters, kernel_size, activation="relu", input_shape=(48, 48, 1) if i == 0 else None))
#         model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(individual["dense_units"], activation="relu"))
#     model.add(Dropout(individual["dropout_rate"]))
#     model.add(Dense(1, activation="sigmoid"))
#     return model

# # Genetic Algorithm
# def genetic_algorithm():
#     population = [{
#         "num_layers": random.choice(LAYERS),
#         "filters": [random.choice(FILTERS) for _ in range(random.choice(LAYERS))],
#         "kernel_size": [random.choice(KERNEL_SIZES) for _ in range(random.choice(LAYERS))],
#         "dense_units": random.choice(DENSE_UNITS),
#         "dropout_rate": random.choice(DROPOUT_RATES)
#     } for _ in range(POPULATION_SIZE)]

#     for generation in range(GENERATIONS):
#         print(f"Generation {generation + 1}")
#         population = sorted(population, key=lambda x: fitness_function(x), reverse=True)
#         best_individual = population[0]
#         print("Best Accuracy:", fitness_function(best_individual))

#         # Crossover and Mutation
#         new_population = [best_individual]
#         for _ in range(POPULATION_SIZE - 1):
#             parent1, parent2 = random.choices(population[:5], k=2)
#             child = {
#                 "num_layers": random.choice([parent1["num_layers"], parent2["num_layers"]]),
#                 "filters": [random.choice([parent1["filters"][i], parent2["filters"][i]]) for i in range(min(len(parent1["filters"]), len(parent2["filters"])))],
#                 "kernel_size": [random.choice([parent1["kernel_size"][i], parent2["kernel_size"][i]]) for i in range(min(len(parent1["kernel_size"]), len(parent2["kernel_size"])))],
#                 "dense_units": random.choice([parent1["dense_units"], parent2["dense_units"]]),
#                 "dropout_rate": random.choice([parent1["dropout_rate"], parent2["dropout_rate"]])
#             }
#             if random.random() < MUTATION_RATE:
#                 child["num_layers"] = random.choice(LAYERS)
#             new_population.append(child)
#         population = new_population

#     return best_individual

# # Run Genetic Algorithm
# best_model_params = genetic_algorithm()
# print("Best Model Parameters:", best_model_params)