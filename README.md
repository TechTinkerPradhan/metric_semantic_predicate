# Metric Semantic Predicate

## Overview
The `Metric Semantic Predicate` package is designed for processing scene graphs, training Bayesian neural networks, and visualizing 3D scenes. It provides tools for data preparation, model training, and interaction with Llama/Ollama for semantic analysis.

## Features
- **Scene Graph Processing**: Load and process scene graphs to extract objects and their relationships.
- **Bayesian Neural Networks**: Train and evaluate Bayesian models for metric and semantic predictions.
- **Visualization**: Visualize 3D scenes and model outputs using Plotly.
- **Llama Integration**: Interact with Llama/Ollama for semantic and predicate extraction.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd metric_semantic_predicate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your scene graph data in the required format.
2. Use the `main.py` script to process data, train models, or visualize results.
   ```bash
   python main.py
   ```

## Project Structure
- `data/`: Placeholder for scene graph data.
- `models/`: Contains model implementations, including Bayesian neural networks and predicate models.
  - `bayesian_nn.py`: Implements Bayesian neural networks.
  - `metric_model.py`: Handles metric-based models.
  - `predicate_model.py`: Processes predicate-based models.
  - `semantic_model.py`: Manages semantic models.
- `scripts/`: Scripts for generating datasets.
  - `generate_function_dataset.py`: Generates function-based datasets.
  - `generate_quaries_dataset.py`: Generates query-based datasets.
  - `simulate_human_dataset.py`: Simulates human interaction datasets.
- `utils/`: Utility scripts for data preparation and processing.
  - `data_preparation.py`: Functions for preparing datasets and processing scene graphs.
  - `process_query.py`: Processes natural language queries.
  - `scene_graph_loader.py`: Loads and processes scene graphs.
  - `visualization.py`: Visualizes 3D scenes and model outputs.
- `tests/`: Unit tests for the package.
  - `test_bayesian_nn.py`: Tests for Bayesian neural networks.
  - `test_data_preparation.py`: Tests for data preparation utilities.
  - `test_models.py`: Tests for model implementations.
  - `test_visualization.py`: Tests for visualization utilities.

## Requirements
See `requirements.txt` for a list of dependencies.

## License
This project is licensed under the MIT License.

<!-- ## Running Tests
To ensure the package is working correctly, run the unit tests:
```bash
python -m unittest discover tests
``` -->

## Examples

### Scene Graph Processing
Use the `scene_graph_loader` module to extract objects and room data from a scene graph:
```python
from metric_semantic_predicate.utils.scene_graph_loader import get_scene_objects_for_room

scene_graph = {
    "room": {1: {"size": (10, 10, 3), "location": (0, 0, 0)}},
    "object": {"obj1": {"class_": "chair", "location": (1, 1, 0), "size": (1, 1, 1), "parent_room": 1}}
}
objects, room_data = get_scene_objects_for_room(scene_graph, 1)
print(objects, room_data)
```

### Bayesian Neural Networks
Train a Bayesian neural network using the `bayesian_nn` module:
```python
from metric_semantic_predicate.models.bayesian_nn import train_bnn_combined
import torch

X_train = torch.rand(100, 6)
Y_train = torch.rand(100, 1)
svi, net = train_bnn_combined(X_train, Y_train, input_dim=6, hidden_dim=32, num_steps=1000)
```

### Query Processing
Process a natural language query using the `process_query` module:
```python
from metric_semantic_predicate.utils.process_query import process_query
from metric_semantic_predicate.utils.scene_graph_loader import SceneObject

scene_objects = [SceneObject(name="vase", position=(1, 1, 1), size=(0.5, 0.5, 0.5))]
query = {"metric": "2 meters", "semantic": "vase", "predicate": "left"}
params = process_query(query, scene_objects)
print(params)
```
