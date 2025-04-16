# dataset/dataset_generator.py

import numpy as np
import pandas as pd
import random

from metric_semantic_predicate.utils.scene_graph_loader import get_scene_objects_for_room
from metric_semantic_predicate.utils.process_query import process_query
from metric_semantic_predicate.utils.pdf_generator import combined_pdf

# Define predicates and distance range
possible_predicates = ["left", "right", "front", "behind", "above", "below"]

def generate_3d_dataset(scene_graph, num_queries=1000):
    all_objects = []
    for obj_id, obj_info in scene_graph["object"].items():
        parent_room = obj_info["parent_room"]
        object_name = obj_info["class_"]
        width, depth, height = obj_info["size"]
        all_objects.append((parent_room, object_name, width, depth, height))

    rows = []

    for _ in range(num_queries):
        parent_room, semantic_label, width, depth, height = random.choice(all_objects)
        dist = random.uniform(1.0, 10.0)
        predicate = random.choice(possible_predicates)

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

    return pd.DataFrame(rows)
