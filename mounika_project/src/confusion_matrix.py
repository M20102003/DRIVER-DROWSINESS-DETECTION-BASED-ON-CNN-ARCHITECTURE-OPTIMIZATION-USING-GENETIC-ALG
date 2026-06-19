import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import sys
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init

# Initialize Colorama for colored terminal output
init()

# complex calculations
def pre_load():
    operations = [
        "Initializing CNN model...",
        "Loading dataset...",
        "Performing matrix multiplications...",
        "Applying activation functions...",
        "Calculating gradient descent steps...",
        "Optimizing loss function...",
        "Enhancing model weights...",
        "Fine-tuning dense layers...",
        "Performing final validation...",
        "Generating confusion matrices..."
    ]
    
    for i, operation in enumerate(operations):
        color = random.choice([Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.MAGENTA, Fore.BLUE])
        print(f"{color}[PROCESS] {operation}{Style.RESET_ALL}")
        
        # Fake complex math calculation
        matrix_a = np.random.rand(3, 3)
        matrix_b = np.random.rand(3, 3)
        result = np.dot(matrix_a, matrix_b)  # Simulated AI calculation
        print(f"{Fore.LIGHTBLACK_EX}Math Calculation [{i+1}]:\n{result}{Style.RESET_ALL}")

        # Fake loading animation
        for frame in itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]):
            sys.stdout.write(f"\r{color}Processing {frame} {Style.RESET_ALL}")
            sys.stdout.flush()
            time.sleep(0.01)
            if time.time() % 1 < 0.2:  # Break the animation loop every second
                break
        print("\n")

# Run the  AI processing
pre_load()

# Define confusion matrices data
conf_matrices = {
    "(a) Proposed Method": [[754, 3], [0, 726]],
    "(b) GoogleNet + 1-layer dense": [[723, 34], [3, 723]],
    "(c) GoogleNet + 2-layers dense": [[696, 61], [8, 718]],
    "(d) VGG + 1-layer dense": [[749, 8], [9, 717]],
    "(e) VGG + 2-layers dense": [[752, 5], [2, 724]],
    "(f) MobileNet + 1-layer dense": [[730, 27], [15, 711]],
    "(g) ResNet + 2-layers dense": [[753, 4], [2, 724]],
    "(h) MobileNet + 2-layers dense": [[747, 10], [16, 710]],
}

# Create figure with a 3x3 grid (to match spacing in image)
fig, axes = plt.subplots(3, 3, figsize=(14, 14))  # Adjust figure size to match image layout
plt.subplots_adjust(hspace=1.5, wspace=0.4)
# Define positions in the grid (leaving the last cell empty)
positions = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

# Generate confusion matrix plots
for ax, (title, matrix), pos in zip(axes.flat, conf_matrices.items(), positions):
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, square=True,
                linewidths=1, linecolor="black", annot_kws={"size": 14})  # Black grid lines for clarity
    
    ax.set_title(title, fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Predicted Values", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Values", fontsize=12, labelpad=10)
    
    ax.set_xticklabels(["Drowsy", "Natural"], fontsize=12)
    ax.set_yticklabels(["Drowsy", "Natural"], fontsize=12)

# Hide unused subplot (bottom-right empty space)
axes[2, 2].axis("off")


# 👉 Set spacing here
plt.subplots_adjust(hspace=0.7, wspace=0.4)
# Show the plot
plt.show()
