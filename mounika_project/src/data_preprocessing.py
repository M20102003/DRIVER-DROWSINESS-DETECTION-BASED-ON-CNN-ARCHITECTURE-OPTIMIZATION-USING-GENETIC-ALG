import os

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import cv2
except ImportError:
    os.system("pip install opencv-python")
    import cv2

try:
    import numpy as np
except ImportError:
    os.system("pip install numpy")
    import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "drowsiness_dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_data")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_and_preprocess_images(folder_path, label):
    print("preprocessing the images......")
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))  
        img = img / 255.0 
        images.append(img)
        labels.append(label)
    print("preprocessing completed..")
    return images, labels


closed_eyes_folder = os.path.join(DATA_DIR, "closeeye")
closed_eyes_images, closed_eyes_labels = load_and_preprocess_images(closed_eyes_folder, 0)


open_eyes_folder = os.path.join(DATA_DIR, "openeye")
open_eyes_images, open_eyes_labels = load_and_preprocess_images(open_eyes_folder, 1)


X = np.array(closed_eyes_images + open_eyes_images)
y = np.array(closed_eyes_labels + open_eyes_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print("Data preprocessing completed. Processed data saved to:", PROCESSED_DIR)