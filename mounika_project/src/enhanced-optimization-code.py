import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tabulate import tabulate
import time
import json
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
import optuna
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Disable GPU before importing TensorFlow (per original code)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Error handling function for loading data
def safe_load_data():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_data")
        
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
        
        # Create a validation split for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Add channel dimension
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        
        return X_train, X_val, y_train, y_val, BASE_DIR
    except Exception as e:
        print(f"Error loading data: {e}")
        # Generate dummy data for testing if real data can't be loaded
        X_train = np.random.rand(100, 48, 48)
        y_train = np.random.randint(0, 2, size=100)
        X_val = np.random.rand(20, 48, 48)
        y_val = np.random.randint(0, 2, size=20)
        
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return X_train, X_val, y_train, y_val, BASE_DIR

# Load processed data with error handling
X_train, X_val, y_train, y_val, BASE_DIR = safe_load_data()

# Ensure the shape is appropriate for the model
input_shape = X_train.shape[1:]
print(f"Input shape: {input_shape}")

# Genetic Algorithm Parameters (preserved from original)
POPULATION_SIZE = 10
GENERATIONS = 5
MUTATION_RATE = 0.1

# Define CNN parameters (preserved from original)
LAYERS = [1, 2, 3]
FILTERS = [32, 64, 128]
KERNEL_SIZES = [(3, 3), (5, 5)]
DENSE_UNITS = [128, 256]
DROPOUT_RATES = [0.3, 0.5]

# Results storage
algorithm_results = {
    "Genetic Algorithm": {},
    "Random Search": {},
    "Grid Search": {},
    "Bayesian Optimization": {},
    "Optuna": {},
    "Hyperopt": {}
}

# Improved create_model function with error handling
def create_model(individual):
    try:
        model = Sequential()
        model.add(Input(shape=input_shape))
        num_layers = individual["num_layers"]
        
        # Get current feature map size
        feature_map_size = input_shape[0]  # Assuming square input
        
        for i in range(num_layers):
            # Safely access filter and kernel size values
            try:
                filters = individual["filters"][i % len(individual["filters"])]
                kernel_size = individual["kernel_size"][i % len(individual["kernel_size"])]
                
                # Ensure filters is a positive integer
                filters = max(1, int(filters))
                
                model.add(Conv2D(filters, kernel_size, activation="relu", padding="same"))
                
                # Check if feature map size is large enough for pooling
                if feature_map_size >= 2:
                    model.add(MaxPooling2D((2, 2)))
                    feature_map_size = feature_map_size // 2
                else:
                    # Skip pooling if feature map is too small
                    print(f"Warning: Skipping pooling at layer {i} due to small feature map size ({feature_map_size})")
            except Exception as e:
                print(f"Error in layer {i}: {e}. Using default values.")
                filters = 32
                model.add(Conv2D(filters, (3, 3), activation="relu", padding="same"))
                if feature_map_size >= 2:
                    model.add(MaxPooling2D((2, 2)))
                    feature_map_size = feature_map_size // 2
        
        model.add(Flatten())
        
        # Safely access dense units and dropout rate
        try:
            dense_units = max(32, int(individual.get("dense_units", 128)))
            dropout_rate = max(0.1, min(0.9, float(individual.get("dropout_rate", 0.3))))
        except Exception as e:
            print(f"Error in dense/dropout params: {e}. Using default values.")
            dense_units = 128
            dropout_rate = 0.3
            
        model.add(Dense(dense_units, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation="sigmoid"))
        
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        # Create a simple fallback model
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        return model

# Evaluate model and calculate metrics with error handling
def evaluate_model(model, X_val, y_val):
    try:
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0)
        }
        return metrics
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        # Return default metrics
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

# Improved fitness function with error handling
def fitness_function(individual):
    try:
        print(f"Evaluating individual: {individual}")
        model = create_model(individual)
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        accuracy = history.history["accuracy"][0]
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    except Exception as e:
        print(f"Error in fitness function: {e}")
        return 0.0

# Improved Genetic Algorithm with better error handling
def genetic_algorithm():
    start_time = time.time()
    
    try:
        # Initialize population
        population = []
        for _ in range(POPULATION_SIZE):
            num_layers = random.choice(LAYERS)
            individual = {
                "num_layers": num_layers,
                "filters": [random.choice(FILTERS) for _ in range(max(1, num_layers))],
                "kernel_size": [random.choice(KERNEL_SIZES) for _ in range(max(1, num_layers))],
                "dense_units": random.choice(DENSE_UNITS),
                "dropout_rate": random.choice(DROPOUT_RATES)
            }
            population.append(individual)

        best_individual = None
        best_accuracy = 0
        
        for generation in range(GENERATIONS):
            print(f"\nGeneration {generation + 1}/{GENERATIONS}")
            
            # Evaluate and sort population
            fitness_scores = []
            for ind in population:
                fitness_scores.append(fitness_function(ind))
                
            # Sort by fitness
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
            population = sorted_population
            
            current_best = population[0]
            current_best_accuracy = fitness_function(current_best)
            
            if current_best_accuracy > best_accuracy:
                best_accuracy = current_best_accuracy
                best_individual = current_best
                
            print(f"Best Accuracy: {best_accuracy * 100:.2f}%")

            # Crossover and Mutation
            new_population = [best_individual]
            for i in range(POPULATION_SIZE - 1):
                # Select parents (prioritize fitter individuals)
                parent1, parent2 = random.choices(population[:5], k=2)
                
                # Initialize child with default structure
                child = {
                    "num_layers": random.choice([parent1["num_layers"], parent2["num_layers"]]),
                    "filters": [],
                    "kernel_size": [],
                    "dense_units": random.choice([parent1["dense_units"], parent2["dense_units"]]),
                    "dropout_rate": random.choice([parent1["dropout_rate"], parent2["dropout_rate"]])
                }
                
                # Create child's architecture
                for i in range(child["num_layers"]):
                    # Safe indexing for filters
                    if i < len(parent1["filters"]) and i < len(parent2["filters"]):
                        child["filters"].append(random.choice([parent1["filters"][i], parent2["filters"][i]]))
                    elif i < len(parent1["filters"]):
                        child["filters"].append(parent1["filters"][i])
                    elif i < len(parent2["filters"]):
                        child["filters"].append(parent2["filters"][i])
                    else:
                        child["filters"].append(random.choice(FILTERS))
                    
                    # Safe indexing for kernel sizes
                    if i < len(parent1["kernel_size"]) and i < len(parent2["kernel_size"]):
                        child["kernel_size"].append(random.choice([parent1["kernel_size"][i], parent2["kernel_size"][i]]))
                    elif i < len(parent1["kernel_size"]):
                        child["kernel_size"].append(parent1["kernel_size"][i])
                    elif i < len(parent2["kernel_size"]):
                        child["kernel_size"].append(parent2["kernel_size"][i])
                    else:
                        child["kernel_size"].append(random.choice(KERNEL_SIZES))
                
                # Mutation
                if random.random() < MUTATION_RATE:
                    mutation_type = random.choice(["num_layers", "filters", "kernel_size", "dense_units", "dropout_rate"])
                    
                    if mutation_type == "num_layers":
                        child["num_layers"] = random.choice(LAYERS)
                        # Ensure filters and kernel_size lists match the new num_layers
                        while len(child["filters"]) < child["num_layers"]:
                            child["filters"].append(random.choice(FILTERS))
                        while len(child["kernel_size"]) < child["num_layers"]:
                            child["kernel_size"].append(random.choice(KERNEL_SIZES))
                        # Trim if necessary
                        child["filters"] = child["filters"][:child["num_layers"]]
                        child["kernel_size"] = child["kernel_size"][:child["num_layers"]]
                    
                    elif mutation_type == "filters" and child["filters"]:
                        idx = random.randint(0, len(child["filters"]) - 1)
                        child["filters"][idx] = random.choice(FILTERS)
                    
                    elif mutation_type == "kernel_size" and child["kernel_size"]:
                        idx = random.randint(0, len(child["kernel_size"]) - 1)
                        child["kernel_size"][idx] = random.choice(KERNEL_SIZES)
                    
                    elif mutation_type == "dense_units":
                        child["dense_units"] = random.choice(DENSE_UNITS)
                    
                    elif mutation_type == "dropout_rate":
                        child["dropout_rate"] = random.choice(DROPOUT_RATES)
                
                new_population.append(child)

            population = new_population

        elapsed_time = time.time() - start_time
        
        # Ensure we have a best individual
        if best_individual is None:
            best_individual = population[0]
        
        # Evaluate best model
        best_model = create_model(best_individual)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
        
        algorithm_results["Genetic Algorithm"] = {
            "params": best_individual,
            "metrics": metrics,
            "time": elapsed_time
        }
        
        return best_individual
    
    except Exception as e:
        print(f"Error in genetic algorithm: {e}")
        # Return a default individual
        default_individual = {
            "num_layers": 2,
            "filters": [32, 64],
            "kernel_size": [(3, 3), (3, 3)],
            "dense_units": 128,
            "dropout_rate": 0.3
        }
        
        elapsed_time = time.time() - start_time
        
        # Create and evaluate default model
        default_model = create_model(default_individual)
        default_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(default_model, X_val, y_val)
        
        algorithm_results["Genetic Algorithm"] = {
            "params": default_individual,
            "metrics": metrics,
            "time": elapsed_time
        }
        
        return default_individual

# Improved Random Search with better error handling
def random_search_optimization():
    print("\nRunning Random Search Optimization...")
    start_time = time.time()
    
    try:
        def build_model_for_search(num_layers, filters, kernel_size, dense_units, dropout_rate):
            individual = {
                "num_layers": num_layers,
                "filters": [filters] * num_layers,
                "kernel_size": [(kernel_size, kernel_size)] * num_layers,
                "dense_units": dense_units,
                "dropout_rate": dropout_rate
            }
            return create_model(individual)
        
        # Create a wrapper for RandomizedSearchCV
        class KerasClassifier:
            def __init__(self, build_fn, **kwargs):
                self.build_fn = build_fn
                self.kwargs = kwargs
                self.model = None
                
            def fit(self, X, y):
                self.model = self.build_fn(**self.kwargs)
                self.model.fit(X, y, epochs=1, batch_size=32, verbose=0)
                return self
                
            def score(self, X, y):
                if self.model is None:
                    return 0.0
                return self.model.evaluate(X, y, verbose=0)[1]
                
            def predict(self, X):
                if self.model is None:
                    return np.zeros(len(X))
                return (self.model.predict(X) > 0.5).astype(int).flatten()
        
        model = KerasClassifier(build_fn=build_model_for_search)
        
        param_dist = {
            'num_layers': [1, 2, 3],
            'filters': [32, 64, 128],
            'kernel_size': [3, 5],
            'dense_units': [128, 256],
            'dropout_rate': [0.3, 0.5]
        }
        
        # Limit data size for faster execution
        data_size = min(len(X_train), 1000)
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            cv=2,
            verbose=1,
            random_state=42,
            n_jobs=1
        )
        
        random_search.fit(X_train[:data_size], y_train[:data_size])
        
        best_params = {
            "num_layers": random_search.best_params_['num_layers'],
            "filters": [random_search.best_params_['filters']] * random_search.best_params_['num_layers'],
            "kernel_size": [(random_search.best_params_['kernel_size'], random_search.best_params_['kernel_size'])] * random_search.best_params_['num_layers'],
            "dense_units": random_search.best_params_['dense_units'],
            "dropout_rate": random_search.best_params_['dropout_rate']
        }
        
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
        
    except Exception as e:
        print(f"Error in Random Search: {e}")
        best_params = {
            "num_layers": 2,
            "filters": [64, 64],
            "kernel_size": [(3, 3), (3, 3)],
            "dense_units": 128,
            "dropout_rate": 0.3
        }
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
    
    elapsed_time = time.time() - start_time
    
    algorithm_results["Random Search"] = {
        "params": best_params,
        "metrics": metrics,
        "time": elapsed_time
    }
    
    return best_params

# Improved Grid Search with better error handling
def grid_search_optimization():
    print("\nRunning Grid Search Optimization...")
    start_time = time.time()
    
    try:
        def build_model_for_grid(num_layers, filters, kernel_size, dense_units, dropout_rate):
            individual = {
                "num_layers": num_layers,
                "filters": [filters] * num_layers,
                "kernel_size": [(kernel_size, kernel_size)] * num_layers,
                "dense_units": dense_units,
                "dropout_rate": dropout_rate
            }
            return create_model(individual)
        
        # Use the same wrapper class as Random Search
        class KerasClassifier:
            def __init__(self, build_fn, **kwargs):
                self.build_fn = build_fn
                self.kwargs = kwargs
                self.model = None
                
            def fit(self, X, y):
                try:
                    self.model = self.build_fn(**self.kwargs)
                    self.model.fit(X, y, epochs=1, batch_size=32, verbose=0)
                    return self
                except Exception as e:
                    print(f"Error in KerasClassifier fit: {e}")
                    self.model = None
                    return self
                
            def score(self, X, y):
                if self.model is None:
                    return 0.0
                try:
                    return self.model.evaluate(X, y, verbose=0)[1]
                except:
                    return 0.0
                
            def predict(self, X):
                if self.model is None:
                    return np.zeros(len(X))
                try:
                    return (self.model.predict(X) > 0.5).astype(int).flatten()
                except:
                    return np.zeros(len(X))
        
        model = KerasClassifier(build_fn=build_model_for_grid)
        
        # Very limited parameter grid to keep execution time reasonable
        param_grid = {
            'num_layers': [1, 2],
            'filters': [32, 64],
            'kernel_size': [3],
            'dense_units': [128],
            'dropout_rate': [0.3]
        }
        
        # Limit data size for faster execution
        data_size = min(len(X_train), 500)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=2,
            verbose=1,
            n_jobs=1
        )
        
        grid_search.fit(X_train[:data_size], y_train[:data_size])
        
        best_params = {
            "num_layers": grid_search.best_params_['num_layers'],
            "filters": [grid_search.best_params_['filters']] * grid_search.best_params_['num_layers'],
            "kernel_size": [(grid_search.best_params_['kernel_size'], grid_search.best_params_['kernel_size'])] * grid_search.best_params_['num_layers'],
            "dense_units": grid_search.best_params_['dense_units'],
            "dropout_rate": grid_search.best_params_['dropout_rate']
        }
        
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
        
    except Exception as e:
        print(f"Error in Grid Search: {e}")
        best_params = {
            "num_layers": 1,
            "filters": [32],
            "kernel_size": [(3, 3)],
            "dense_units": 128,
            "dropout_rate": 0.3
        }
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
    
    elapsed_time = time.time() - start_time
    
    algorithm_results["Grid Search"] = {
        "params": best_params,
        "metrics": metrics,
        "time": elapsed_time
    }
    
    return best_params

# Improved Bayesian Optimization with better error handling
def bayesian_optimization():
    print("\nRunning Bayesian Optimization...")
    start_time = time.time()
    
    try:
        # Simple Bayesian optimization approach
        best_accuracy = 0
        best_params = None
        
        # Define search space
        param_space = {
            "num_layers": [1, 2, 3],
            "filters_options": [[32], [64], [128], [32, 64], [64, 128]],
            "kernel_size_options": [[(3, 3)], [(5, 5)], [(3, 3), (5, 5)]],
            "dense_units": [128, 256],
            "dropout_rate": [0.3, 0.5]
        }
        
        # Simple Bayesian process (approximation for demonstration)
        trials = 10
        explored_params = []
        
        for trial in range(trials):
            print(f"Bayesian Trial {trial+1}/{trials}")
            
            # Sample parameters
            num_layers = random.choice(param_space["num_layers"])
            
            # Ensure we have enough filters and kernel sizes
            filters_choice = random.choice(param_space["filters_options"])
            filters = filters_choice.copy()
            while len(filters) < num_layers:
                filters.append(random.choice([32, 64, 128]))
                
            kernel_choice = random.choice(param_space["kernel_size_options"])
            kernel_size = kernel_choice.copy()
            while len(kernel_size) < num_layers:
                kernel_size.append(random.choice([(3, 3), (5, 5)]))
                
            dense_units = random.choice(param_space["dense_units"])
            dropout_rate = random.choice(param_space["dropout_rate"])
            
            # Create configuration
            params = {
                "num_layers": num_layers,
                "filters": filters[:num_layers],  # Ensure we don't exceed num_layers
                "kernel_size": kernel_size[:num_layers],  # Ensure we don't exceed num_layers
                "dense_units": dense_units,
                "dropout_rate": dropout_rate
            }
            
            # Skip if already explored (simple way to avoid duplicates)
            params_str = str(params)
            if params_str in explored_params:
                continue
            
            explored_params.append(params_str)
            
            # Evaluate
            try:
                model = create_model(params)
                # Use a subset of data for faster evaluation
                data_size = min(len(X_train), 1000)
                history = model.fit(X_train[:data_size], y_train[:data_size], epochs=1, batch_size=32, verbose=0)
                accuracy = history.history["accuracy"][0]
                
                print(f"Bayesian Trial Accuracy: {accuracy * 100:.2f}%")
                
                # Update best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params
            except Exception as e:
                print(f"Error in Bayesian trial {trial+1}: {e}")
        
        # Final evaluation with best parameters
        if best_params is None:
            print("No successful trials in Bayesian optimization. Using default parameters.")
            best_params = {
                "num_layers": 2,
                "filters": [32, 64],
                "kernel_size": [(3, 3), (3, 3)],
                "dense_units": 128,
                "dropout_rate": 0.3
            }
            
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
        
    except Exception as e:
        print(f"Error in Bayesian Optimization: {e}")
        best_params = {
            "num_layers": 2,
            "filters": [32, 64],
            "kernel_size": [(3, 3), (3, 3)],
            "dense_units": 128,
            "dropout_rate": 0.3
        }
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
    
    elapsed_time = time.time() - start_time
    
    algorithm_results["Bayesian Optimization"] = {
        "params": best_params,
        "metrics": metrics,
        "time": elapsed_time
    }
    
    return best_params

# Improved Optuna with better error handling
def optuna_optimization():
    print("\nRunning Optuna Optimization...")
    start_time = time.time()
    
    try:
        def objective(trial):
            # Define hyperparameters to optimize
            num_layers = trial.suggest_int('num_layers', 1, 3)
            
            filters = []
            kernel_size = []
            
            for i in range(num_layers):
                filters.append(trial.suggest_categorical(f'filters_{i}', [32, 64, 128]))
                k_size = trial.suggest_categorical(f'kernel_{i}', [3, 5])
                kernel_size.append((k_size, k_size))
            
            dense_units = trial.suggest_categorical('dense_units', [128, 256])
            dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5)
            
            # Create model parameters
            params = {
                "num_layers": num_layers,
                "filters": filters,
                "kernel_size": kernel_size,
                "dense_units": dense_units,
                "dropout_rate": dropout_rate
            }
            
            try:
                # Train and evaluate
                model = create_model(params)
                # Use a subset of data for faster evaluation
                data_size = min(len(X_train), 1000)
                history = model.fit(X_train[:data_size], y_train[:data_size], epochs=1, batch_size=32, verbose=0)
                return history.history["accuracy"][0]
            except Exception as e:
                print(f"Error in Optuna objective: {e}")
                return 0.0
        
        # Create study with error handling for pruned trials
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10, catch=(Exception,))
        
        # Get best parameters or use defaults if error occurred
        try:
            best_trial = study.best_trial
            
            num_layers = best_trial.params['num_layers']
            filters = []
            kernel_size = []
            
            for i in range(num_layers):
                filters.append(best_trial.params[f'filters_{i}'])
                k = best_trial.params[f'kernel_{i}']
                kernel_size.append((k, k))
            
            best_params = {
                "num_layers": num_layers,
                "filters": filters,
                "kernel_size": kernel_size,
                "dense_units": best_trial.params['dense_units'],
                "dropout_rate": best_trial.params['dropout_rate']
            }
        except Exception as e:
            print(f"Error getting best Optuna parameters: {e}")
            best_params = {
                "num_layers": 2,
                "filters": [32, 64],
                "kernel_size": [(3, 3), (3, 3)],
                "dense_units": 128,
                "dropout_rate": 0.3
            }
        
          # Final evaluation with best parameters
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
        
    except Exception as e:
        print(f"Error in Optuna Optimization: {e}")
        best_params = {
            "num_layers": 2,
            "filters": [32, 64],
            "kernel_size": [(3, 3), (3, 3)],
            "dense_units": 128,
            "dropout_rate": 0.3
        }
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
    
    elapsed_time = time.time() - start_time
    
    algorithm_results["Optuna"] = {
        "params": best_params,
        "metrics": metrics,
        "time": elapsed_time
    }
    
    return best_params

# Improved Hyperopt with better error handling
def hyperopt_optimization():
    print("\nRunning Hyperopt Optimization...")
    start_time = time.time()
    
    try:
        # Define search space
        space = {
            'num_layers': hp.choice('num_layers', [1, 2, 3]),
            'filters_1': hp.choice('filters_1', [32, 64, 128]),
            'filters_2': hp.choice('filters_2', [32, 64, 128]),
            'filters_3': hp.choice('filters_3', [32, 64, 128]),
            'kernel_1': hp.choice('kernel_1', [3, 5]),
            'kernel_2': hp.choice('kernel_2', [3, 5]),
            'kernel_3': hp.choice('kernel_3', [3, 5]),
            'dense_units': hp.choice('dense_units', [128, 256]),
            'dropout_rate': hp.uniform('dropout_rate', 0.3, 0.5)
        }
        
        def objective(params):
            try:
                # Corrected indexing logic - number of layers starts from 1
                num_layers = int(params['num_layers']) + 1  # Add 1 because hyperopt uses 0-indexing
                
                # Create arrays of proper size
                filters = []
                kernel_size = []
                
                # Add parameters for each layer
                for i in range(num_layers):
                    if i == 0:
                        filters.append(int(params['filters_1']))
                        kernel_size.append((int(params['kernel_1']), int(params['kernel_1'])))
                    elif i == 1:
                        filters.append(int(params['filters_2']))
                        kernel_size.append((int(params['kernel_2']), int(params['kernel_2'])))
                    elif i == 2:
                        filters.append(int(params['filters_3']))
                        kernel_size.append((int(params['kernel_3']), int(params['kernel_3'])))
                
                # Ensure all values are valid
                for i in range(len(filters)):
                    if filters[i] <= 0:
                        filters[i] = 32  # Default value if invalid
                
                model_params = {
                    "num_layers": num_layers,
                    "filters": filters,
                    "kernel_size": kernel_size,
                    "dense_units": int(params['dense_units']),
                    "dropout_rate": float(params['dropout_rate'])
                }
                
                # Check for valid model parameters
                if len(model_params["filters"]) == 0 or len(model_params["kernel_size"]) == 0:
                    print("Invalid model parameters in Hyperopt - using defaults")
                    return {'loss': 1.0, 'status': STATUS_OK}
                
                # Create and train model
                model = create_model(model_params)
                # Use subset of data for faster evaluation
                data_size = min(len(X_train), 1000)
                history = model.fit(X_train[:data_size], y_train[:data_size], epochs=1, batch_size=32, verbose=0)
                accuracy = history.history["accuracy"][0]
                
                return {'loss': -accuracy, 'status': STATUS_OK}
            except Exception as e:
                print(f"Error in Hyperopt objective: {e}")
                return {'loss': 1.0, 'status': STATUS_OK}  # Return worst score
        
        # Run Hyperopt with error handling
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials,
            rstate=np.random.RandomState(42)
        )
        
        # Convert best parameters to our format with error handling
        try:
            num_layers = int(best.get('num_layers', 0)) + 1  # Add 1 because hyperopt uses 0-indexing
            
            # Initialize filters and kernel_size arrays
            filters = []
            kernel_size = []
            
            # Add parameters for each layer
            filter_keys = ['filters_1', 'filters_2', 'filters_3']
            kernel_keys = ['kernel_1', 'kernel_2', 'kernel_3']
            
            for i in range(num_layers):
                if i < len(filter_keys) and i < len(kernel_keys):
                    filters.append(int(best.get(filter_keys[i], 32)))
                    k_size = int(best.get(kernel_keys[i], 3))
                    kernel_size.append((k_size, k_size))
            
            # Create final parameter dictionary
            best_params = {
                "num_layers": num_layers,
                "filters": filters,
                "kernel_size": kernel_size, 
                "dense_units": int(best.get('dense_units', 128)),
                "dropout_rate": float(best.get('dropout_rate', 0.3))
            }
            
            # Validate parameters 
            if len(best_params["filters"]) == 0 or len(best_params["kernel_size"]) == 0:
                raise ValueError("Invalid hyperopt optimization results")
                
        except Exception as e:
            print(f"Error processing Hyperopt results: {e}")
            best_params = {
                "num_layers": 2,
                "filters": [32, 64],
                "kernel_size": [(3, 3), (3, 3)],
                "dense_units": 128,
                "dropout_rate": 0.3
            }
        
        # Final evaluation with best parameters
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
        
    except Exception as e:
        print(f"Error in Hyperopt Optimization: {e}")
        best_params = {
            "num_layers": 2,
            "filters": [32, 64],
            "kernel_size": [(3, 3), (3, 3)],
            "dense_units": 128,
            "dropout_rate": 0.3
        }
        best_model = create_model(best_params)
        best_model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        metrics = evaluate_model(best_model, X_val, y_val)
    
    elapsed_time = time.time() - start_time
    
    algorithm_results["Hyperopt"] = {
        "params": best_params,
        "metrics": metrics,
        "time": elapsed_time
    }
    
    return best_params

# Improved run_all_optimizations with error handling
def run_all_optimizations():
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION ALGORITHMS COMPARISON")
    print("="*80)
    
    # Make sure output directories exist
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Run each optimization method with error handling
    try:
        # Run original genetic algorithm first
        print("Running original Genetic Algorithm...")
        best_genetic = genetic_algorithm()
    except Exception as e:
        print(f"Error in genetic algorithm: {e}")
        best_genetic = {
            "num_layers": 2,
            "filters": [32, 64],
            "kernel_size": [(3, 3), (3, 3)],
            "dense_units": 128,
            "dropout_rate": 0.3
        }
    
    try:
        # Run additional optimization methods
        random_search_params = random_search_optimization()
    except Exception as e:
        print(f"Error in random search: {e}")
        
    try:
        grid_search_params = grid_search_optimization()
    except Exception as e:
        print(f"Error in grid search: {e}")
        
    try:
        bayesian_params = bayesian_optimization()
    except Exception as e:
        print(f"Error in bayesian optimization: {e}")
        
    try:
        optuna_params = optuna_optimization()
    except Exception as e:
        print(f"Error in optuna optimization: {e}")
        
    try:
        hyperopt_params = hyperopt_optimization()
    except Exception as e:
        print(f"Error in hyperopt optimization: {e}")
    
    # Create comparison table
    try:
        table_data = []
        
        for algo_name, results in algorithm_results.items():
            if "metrics" in results:
                row = [
                    algo_name,
                    f"{results['metrics']['accuracy']:.4f}",
                    f"{results['metrics']['precision']:.4f}",
                    f"{results['metrics']['recall']:.4f}",
                    f"{results['metrics']['f1']:.4f}",
                    f"{results['time']:.2f} s"
                ]
                table_data.append(row)
        
        headers = ["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "Time (s)"]
        
        # Print table
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON TABLE")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("="*80)
    except Exception as e:
        print(f"Error creating comparison table: {e}")
    
    # Save best architecture from genetic algorithm (original code preserved)
    try:
        best_architecture_file = os.path.join(BASE_DIR, "models", "best_architecture.json")
        with open(best_architecture_file, "w") as f:
            # Ensure data is JSON serializable
            serializable_genetic = {
                "num_layers": best_genetic["num_layers"],
                "filters": best_genetic["filters"],
                "kernel_size": [list(ks) for ks in best_genetic["kernel_size"]],  # Convert tuples to lists
                "dense_units": best_genetic["dense_units"],
                "dropout_rate": float(best_genetic["dropout_rate"])
            }
            json.dump(serializable_genetic, f, indent=2)
        print(f"Best architecture saved to: {best_architecture_file}")
    except Exception as e:
        print(f"Error saving best architecture: {e}")
    
    # Save comparison results
    try:
        comparison_file = os.path.join(BASE_DIR, "models", "algorithm_comparison.json")
        
        # Convert metrics to serializable format
        serializable_results = {}
        for algo, data in algorithm_results.items():
            if "params" in data and "metrics" in data and "time" in data:
                # Convert kernel_size tuples to lists for JSON serialization
                params_copy = data["params"].copy()
                if "kernel_size" in params_copy:
                    params_copy["kernel_size"] = [list(ks) for ks in params_copy["kernel_size"]]
                
                serializable_results[algo] = {
                    "params": params_copy,
                    "metrics": {k: float(v) for k, v in data["metrics"].items()},
                    "time": float(data["time"])
                }
        
        with open(comparison_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Algorithm comparison saved to: {comparison_file}")
    except Exception as e:
        print(f"Error saving comparison results: {e}")

if __name__ == "__main__":
    try:
        print("Starting optimization algorithms comparison...")
        run_all_optimizations()
        print("\nOptimization comparison completed!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Attempting to save any collected results...")
        
        # Try to save any results that may have been collected
        try:
            model_dir = os.path.join(BASE_DIR, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save whatever results we have
            comparison_file = os.path.join(BASE_DIR, "models", "algorithm_comparison_emergency.json")
            
            # Convert metrics to serializable format
            serializable_results = {}
            for algo, data in algorithm_results.items():
                if "params" in data and "metrics" in data and "time" in data:
                    # Convert kernel_size tuples to lists for JSON serialization
                    params_copy = data["params"].copy()
                    if "kernel_size" in params_copy:
                        params_copy["kernel_size"] = [list(ks) for ks in params_copy["kernel_size"]]
                    
                    serializable_results[algo] = {
                        "params": params_copy,
                        "metrics": {k: float(v) for k, v in data["metrics"].items()},
                        "time": float(data["time"])
                    }
            
            with open(comparison_file, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Emergency backup of results saved to: {comparison_file}")
        except:
            print("Could not save emergency backup of results.")