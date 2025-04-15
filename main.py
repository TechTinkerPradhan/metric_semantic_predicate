# Main entry point for the Metric Semantic Predicate package

from metric_semantic_predicate.utils.data_preparation import prepare_dataset
from metric_semantic_predicate.models.bayesian_nn import BayesianNNcombined
import numpy as np
import os
import zipfile

# Example usage of the package
if __name__ == "__main__":
    print("Welcome to the Metric Semantic Predicate package!")

    # Directory containing the scene graph files
    directory_path = "/home/artemis/project/scene_graph/data/automated_graph"

    # Iterate over all .npz files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".npz"):
            file_path = os.path.join(directory_path, file_name)

            # Extract and load .npy files from the .npz archive
            with zipfile.ZipFile(file_path, 'r') as archive:
                for npy_file in archive.namelist():
                    if npy_file.endswith(".npy"):
                        with archive.open(npy_file) as npy_data:
                            scene_graph = np.load(npy_data, allow_pickle=True).item()
                            
