import graphviz

architecture = {
    "num_layers": 3,
    "filters": [32, 128, 128],
    "kernel_size": [[3, 3], [3, 3], [5, 5]],
    "dense_units": 256,
    "dropout_rate": 0.3
}

dot = graphviz.Digraph(comment='CNN Architecture', format='png')
dot.attr('graph', rankdir='LR')  # Left to Right layout

# Input Layer (example)
dot.node('input', 'Input', shape='rectangle')

# Convolutional Layers
prev_node = 'input'
for i in range(architecture['num_layers']):
    layer_name = f'conv{i+1}'
    label = f"Conv2D\nFilters: {architecture['filters'][i]}\nKernel: {architecture['kernel_size'][i]}"
    dot.node(layer_name, label, shape='rectangle')
    dot.edge(prev_node, layer_name)
    prev_node = layer_name

    # Add MaxPooling (example - adjust as needed)
    pool_name = f'pool{i+1}'
    dot.node(pool_name, 'MaxPooling2D', shape='ellipse')  # Or another shape
    dot.edge(prev_node, pool_name)
    prev_node = pool_name

# Flatten Layer
dot.node('flatten', 'Flatten', shape='ellipse')
dot.edge(prev_node, 'flatten')
prev_node = 'flatten'

# Dropout Layer
dot.node('dropout', f'Dropout\nRate: {architecture["dropout_rate"]}', shape='ellipse')
dot.edge(prev_node, 'dropout')
prev_node = 'dropout'

# Dense Layer
dot.node('dense', f'Dense\nUnits: {architecture["dense_units"]}', shape='rectangle')
dot.edge(prev_node, 'dense')

# Output Layer (example - adjust for your task)
dot.node('output', 'Output', shape='rectangle')
dot.edge('dense', 'output')

dot.render('cnn_architecture', view=True)  # Save and display the graph