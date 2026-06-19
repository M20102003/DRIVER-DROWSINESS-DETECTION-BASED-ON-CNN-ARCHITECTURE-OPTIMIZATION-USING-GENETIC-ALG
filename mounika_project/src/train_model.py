
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Ensure GPU usage if available
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = r"C:\Users\cnmou\Desktop\mounika_project\data\processed_data"
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load dataset
try:
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    # Ensure correct shape
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
except Exception as e:
    raise FileNotFoundError(f"Error loading dataset: {e}")

# Load best architecture
try:
    with open(os.path.join(MODELS_DIR, "best_architecture.json"), "r") as f:
        best_architecture = json.load(f)
except Exception as e:
    raise FileNotFoundError("Best architecture file not found! Run the genetic algorithm first.")

# Build the model
model = Sequential()
model.add(Input(shape=(48, 48, 1)))  # Input layer

for i in range(best_architecture["num_layers"]):
    model.add(Conv2D(best_architecture["filters"][i], best_architecture["kernel_size"][i], activation="relu"))
    model.add(BatchNormalization())  # Stabilize training
    model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(best_architecture["dense_units"], activation="relu"))
model.add(Dropout(best_architecture["dropout_rate"]))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Save model
model_path = os.path.join(MODELS_DIR, "cnn_model.h5")
model.save(model_path)
print(f"Model saved at {model_path}")

# Evaluate model
y_pred = (model.predict(X_val) > 0.5).astype("int32")
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)

# Print scores
print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")

# Save model performance to a file
results_path = os.path.join(MODELS_DIR, "model_performance.json")
results = {"accuracy": accuracy, "f1_score": f1, "recall": recall, "precision": precision}
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Model performance metrics saved at {results_path}")
