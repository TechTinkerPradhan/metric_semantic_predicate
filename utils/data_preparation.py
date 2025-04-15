# Added functions from the notebook

import numpy as np
import pandas as pd
import random
from metric_semantic_predicate.utils.process_query import process_query
from metric_semantic_predicate.utils.pdf_generator import combined_pdf


class SceneObject:
    def __init__(self, name, position, size):
        self.name = name
        self.position = position  # (x, y, z)
        self.size = size          # (width, depth, height)

    def __repr__(self):
        return f"SceneObject(name={self.name}, position={self.position}, size={self.size})"

def get_scene_objects_for_room(scene_graph, room_id):
    """
    Returns a tuple: (list_of_objects_in_room, (room_position, room_size))
    """
    room_data = scene_graph["room"].get(room_id)
    if not room_data:
        print(f"Room {room_id} not found in scene_graph['room'].")
        return [], ((0, 0, 0), (0, 0, 0))

    (rwidth, rdepth, rheight) = room_data["size"]
    (rx, ry, rz) = room_data["location"]

    scene_objects = []
    for _, obj_info in scene_graph["object"].items():
        if obj_info["parent_room"] == room_id:
            obj = SceneObject(
                obj_info["class_"],
                tuple(obj_info["location"]),
                tuple(obj_info["size"])
            )
            scene_objects.append(obj)

    return scene_objects, ((rx, ry, rz), (rwidth, rdepth, rheight))

def generate_3d_dataset(scene_graph, num_queries=1000):
    """
    Generates a dataset of randomly constructed queries and their resulting
    3D parameters from process_query, including room properties and P_combined.

    Parameters
    ----------
    scene_graph : dict
        Your scene graph with "room" and "object" keys, etc.
    num_queries : int
        Number of distinct queries to generate.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
          [query, semantic, metric, predicate,
           width, height, depth,
           mu_x, mu_y, mu_z, sigma_s,
           x0, y0, z0, d0, sigma_m,
           theta0, phi0, kappa,
           room_x, room_y, room_z, room_width, room_height, room_depth, parent_room,
           P_combined]
    """
    all_objects = []
    for obj_id, obj_info in scene_graph["object"].items():
        parent_room = obj_info["parent_room"]
        object_name = obj_info["class_"]
        width, depth, height = obj_info["size"]
        all_objects.append((parent_room, object_name, width, depth, height))

    rows = []

    for i in range(num_queries):
        parent_room, semantic_label, width, depth, height = random.choice(all_objects)
        dist = random.uniform(1.0, 10.0)
        predicate = random.choice(["left", "right", "behind", "front", "above", "below"])

        query = {
            "metric": f"{dist:.2f} meters",
            "semantic": semantic_label,
            "predicate": predicate
        }

        scene_objs, (room_pos, room_size) = get_scene_objects_for_room(scene_graph, parent_room)

        try:
            params = process_query(query, scene_objs)
        except Exception as e:
            print(f"Failed to process query {query}: {e}")
            continue

        room_x, room_y, room_z = room_pos
        room_width, room_depth, room_height = room_size

        p_combined = combined_pdf(params["x0"], params["y0"], params["z0"], params)

        row_data = {
            "query": f"{query['metric']} {query['predicate']} {query['semantic']}",
            "semantic": semantic_label,
            "metric": dist,
            "predicate": predicate,
            "width": width,
            "height": height,
            "depth": depth,
            "mu_x": params["mu_x"],
            "mu_y": params["mu_y"],
            "mu_z": params["mu_z"],
            "sigma_s": params["sigma_s"],
            "x0": params["x0"],
            "y0": params["y0"],
            "z0": params["z0"],
            "d0": params["d0"],
            "sigma_m": params["sigma_m"],
            "theta0": params["theta0"],
            "phi0": params["phi0"],
            "kappa": params["kappa"],
            "room_x": room_x,
            "room_y": room_y,
            "room_z": room_z,
            "room_width": room_width,
            "room_height": room_height,
            "room_depth": room_depth,
            "parent_room": parent_room,
            "P_combined": p_combined
        }
        rows.append(row_data)

    df = pd.DataFrame(rows)
    return df