import numpy as np
import matplotlib.pyplot as plt
import json

# Load the architecture data
architecture = {
    "num_layers": 3,
    "filters": [32, 128, 128],
    "kernel_size": [[3, 3], [3, 3], [5, 5]],
    "dense_units": 256,
    "dropout_rate": 0.3
}

# Define input shape (assuming for calculation purposes)
input_shape = (32, 32, 3)  # Example input shape (height, width, channels)

# Calculate parameters for each layer
parameters = []
layer_names = []
input_channels = input_shape[2]

# Calculate CNN parameters
for i in range(architecture["num_layers"]):
    # Conv layer parameters: kernel_h * kernel_w * input_channels * output_filters + bias
    kernel_h, kernel_w = architecture["kernel_size"][i]
    filters = architecture["filters"][i]
    
    conv_params = kernel_h * kernel_w * input_channels * filters + filters
    parameters.append(conv_params)
    layer_names.append(f"Conv2D-{i+1}")
    
    # Update input channels for next layer
    input_channels = filters

# Calculate Dense layer parameters
# Assuming we have a flattening layer before dense
# For simplicity, let's assume the feature map size is reduced by half after each conv layer
h, w = input_shape[0], input_shape[1]
for _ in range(architecture["num_layers"]):
    h = h // 2
    w = w // 2

flattened_size = h * w * architecture["filters"][-1]
dense_params = flattened_size * architecture["dense_units"] + architecture["dense_units"]
parameters.append(dense_params)
layer_names.append("Dense")

# Create a visualization of the parameter count
plt.figure(figsize=(12, 6))

# Bar chart
plt.subplot(1, 2, 1)
bars = plt.bar(layer_names, parameters, color='skyblue')
plt.title('Parameter Count per Layer')
plt.ylabel('Number of Parameters')
plt.xticks(rotation=45)

# Add parameter count annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', rotation=0)

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(parameters, labels=layer_names, autopct='%1.1f%%', startangle=90, 
        colors=plt.cm.viridis(np.linspace(0, 0.9, len(parameters))))
plt.title('Parameter Distribution')

# Add total parameter count as text
total_params = sum(parameters)
plt.figtext(0.5, 0.01, f'Total Parameters: {total_params:,}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('parameter_count_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Print parameter counts for each layer
print("Parameter counts by layer:")
for name, count in zip(layer_names, parameters):
    print(f"{name}: {count:,} parameters")
print(f"Total: {total_params:,} parameters")