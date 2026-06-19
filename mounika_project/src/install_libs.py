import os

# Function to install required libraries
def install_libs():
    required_libs = {
        "opencv-python": "cv2",
        "numpy": "numpy",
        "tensorflow": "tensorflow",
        "playsound": "playsound",
        "scikit-learn": "sklearn"
    }

    for lib, module in required_libs.items():
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {lib}...")
            os.system(f"pip install {lib}")

# Install missing libraries
install_libs()

# Import the installed libraries
import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
