import os
import zipfile
import numpy as np

def get_graph_directory():
    directory_path = os.path.join(os.path.dirname(__file__), "..","..","automated_graph")
    return os.path.abspath(directory_path)

def load_all_scene_graphs():
    return _load_scene_graphs_internal()

def load_scene_graphs_by_name(requested_names):
    return _load_scene_graphs_internal(filter_names=set(requested_names), match_partial=True)


def _load_scene_graphs_internal(filter_names=None, match_partial=False):
    directory_path = get_graph_directory()
    scene_graphs = {}

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".npz"):
            house_name = os.path.splitext(file_name)[0]

            # Match logic
            if filter_names:
                if match_partial:
                    if not any(name.lower() in house_name.lower() for name in filter_names):
                        continue
                else:
                    if house_name not in filter_names:
                        continue

            file_path = os.path.join(directory_path, file_name)
            with zipfile.ZipFile(file_path, 'r') as archive:
                for npy_file in archive.namelist():
                    if npy_file.endswith(".npy"):
                        with archive.open(npy_file) as npy_data:
                            scene_graph = np.load(npy_data, allow_pickle=True).item()
                            scene_graphs[house_name] = scene_graph

    return scene_graphs



