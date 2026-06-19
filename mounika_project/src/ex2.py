import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json

# Load the architecture data
architecture = {
    "num_layers": 3,
    "filters": [32, 128, 128],
    "kernel_size": [[3, 3], [3, 3], [5, 5]],
    "dense_units": 256,
    "dropout_rate": 0.3
}

# Define input shape
input_shape = (32, 32, 3)  # height, width, channels

# Create figure
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Define colors
colors = plt.cm.viridis(np.linspace(0.1, 0.9, architecture["num_layers"] + 2))

# Track current shape and position
current_height, current_width, current_channels = input_shape
y_position = 0
layer_spacing = 0.8  # Vertical spacing between layers
max_width = 8  # Maximum width for visualization scaling

# Function to draw a 3D box as a parallelogram
def draw_3d_box(y_pos, width, height, depth, color, alpha=0.7, label=None, params=None):
    # Scale dimensions for better visualization
    width_scaled = width / 8 * max_width
    height_scaled = height / 8 * max_width
    depth_scaled = depth / 16 * max_width
    
    # Create parallelogram points (pseudo-3D effect)
    shift = max_width / 4  # For 3D effect
    points = np.array([
        [0, y_pos],  # bottom-left
        [width_scaled, y_pos],  # bottom-right
        [width_scaled + shift, y_pos + depth_scaled],  # top-right
        [shift, y_pos + depth_scaled],  # top-left
    ])
    
    # Draw the shape
    polygon = Polygon(points, closed=True, facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(polygon)
    
    # Add label in the middle of the parallelogram
    if label:
        label_text = f"{label}\n{width}×{height}×{depth}"
        if params:
            label_text += f"\n{params:,} params"
        ax.text(width_scaled/2 + shift/2, y_pos + depth_scaled/2, label_text, 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    return y_pos + depth_scaled + layer_spacing

# Draw input layer
y_position = draw_3d_box(y_position, current_width, current_height, current_channels, 
                         colors[0], label="Input")

# Calculate and draw convolutional layers
total_params = 0
for i in range(architecture["num_layers"]):
    # Calculate parameters
    kernel_h, kernel_w = architecture["kernel_size"][i]
    filters = architecture["filters"][i]
    conv_params = kernel_h * kernel_w * current_channels * filters + filters
    total_params += conv_params
    
    # Update dimensions (simplified - assuming stride=1, padding='same' for width/height)
    # In a real network, you'd calculate actual output dimensions based on stride and padding
    current_height = current_height // 2  # Assuming pooling reduces size by half
    current_width = current_width // 2
    current_channels = filters
    
    y_position = draw_3d_box(y_position, current_width, current_height, current_channels, 
                           colors[i+1], label=f"Conv2D-{i+1}", params=conv_params)

# Calculate flattened size
flattened_size = current_height * current_width * current_channels
dense_params = flattened_size * architecture["dense_units"] + architecture["dense_units"]
total_params += dense_params

# Draw flattening layer (transitional representation)
y_position = draw_3d_box(y_position, 1, 1, flattened_size, colors[-2], 
                       label="Flatten", params=0)

# Draw dense layer
y_position = draw_3d_box(y_position, 1, 1, architecture["dense_units"], colors[-1], 
                       label="Dense", params=dense_params)

# Add dropout indicator
ax.text(max_width/2, y_position + 0.2, f"Dropout ({architecture['dropout_rate']})", 
        ha='center', fontsize=9, fontweight='bold')

# Add total parameters
ax.text(max_width/2, y_position + 1, f"Total Parameters: {total_params:,}", 
        ha='center', fontsize=12, fontweight='bold')

# Set axis properties
ax.set_xlim(-1, max_width + 2)
ax.set_ylim(0, y_position + 2)
ax.axis('off')

# Add title
plt.title('CNN Architecture Visualization', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('cnn_architecture_visualization.png', dpi=300, bbox_inches='tight')
plt.show()