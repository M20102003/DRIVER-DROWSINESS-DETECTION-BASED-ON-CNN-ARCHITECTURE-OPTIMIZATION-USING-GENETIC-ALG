# Driver Drowsiness Detection Based on CNN Architecture Optimization Using Genetic Algorithm

This project automates the design of highly efficient Convolutional Neural Network (CNN) architectures for real-time driver drowsiness detection. By implementing a Genetic Algorithm (GA), the system evolves network structures (tuning filters, layer configurations, and activation functions) to balance high classification accuracy with low computational latency.

## 🚀 Key Features
* **Automated Architecture Search**: Uses a Genetic Algorithm to discover optimal CNN configurations without manual trial-and-error.
* **Real-Time Detection**: Features optimized pipeline scripts for real-time inference using webcam video streams.
* **Modular Pipeline**: Cleanly separated folders for dataset processing, GA evolutionary search routines, and evaluation metrics.

## 📋 Prerequisites
Before running the project, ensure you have the following software installed on your machine:
* Python 3.8 to 3.10
* pip (Python Package Installer)
* OpenCV supported hardware (Webcam)

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd DRIVER-DROWSINESS-DETECTION-BASED-ON-CNN-ARCHITECTURE-OPTIMIZATION-USING-GENETIC-ALGO
   ```

2. **Install required libraries:**
   ```bash
   pip install -r mounika_project/src/install_libs.py
   ```
   *(Or manually install core packages: `pip install tensorflow opencv-python numpy matplotlib`)*

3. **Dataset Preparation:**
   * Place your training images inside the local `data/` folder directory before starting.

## 💻 How to Run the Project

### 1. Run Architecture Optimization (Genetic Algorithm)
To start evolving and searching for the best-performing CNN architecture configuration, execute:
```bash
python mounika_project/src/genetic_algorithm.py
```

### 2. Train the Best Evolved Model
Once the optimal hyperparameter structure is selected by the GA search, run training using:
```bash
python mounika_project/src/train_model.py
```

### 3. Run Real-Time Drowsiness Detection
To boot up your webcam and deploy your finalized trained model for live detection tracking, run:
```bash
python mounika_project/src/real_time_detection.py
```

## 📁 Core Codebase Layout
* `mounika_project/src/` — Contains algorithmic modules (`genetic_algorithm.py`, `data_preprocessing.py`, `train_model.py`).
* `mounika_project/models/` — Holds exported architecture topologies (`best_architecture.json`) and trained weight matrices (`cnn_model.h5`).
*
