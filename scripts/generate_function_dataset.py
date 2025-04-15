import numpy as np
import pandas as pd
import random
from metric_semantic_predicate.utils.data_preparation import get_scene_objects_for_room
from metric_semantic_predicate.utils.process_query import process_query
from metric_semantic_predicate.utils.pdf_generator import combined_pdf


# Example list of predicates for 3D (Llama-style strings)
# Make sure these match how your process_query interprets them, e.g. "left of" => (theta0=?, phi0=?)
possible_predicates = ["left", "right", "front", "behind", "above", "below"]

# Possible distances
distances = np.linspace(1.0, 5.0, 20)  # e.g. 20 distinct distances from 1 to 5 meters

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
    # Gather all (room_id, object_name, size) pairs from the scene graph
    all_objects = []
    for obj_id, obj_info in scene_graph["object"].items():
        parent_room = obj_info["parent_room"]
        object_name = obj_info["class_"]
        width, depth, height = obj_info["size"]
        all_objects.append((parent_room, object_name, width, depth, height))

    rows = []

    for i in range(num_queries):
        # Randomly select an object and its parent room
        parent_room, semantic_label, width, depth, height = random.choice(all_objects)
        
        # Randomly pick a metric distance
        dist = random.uniform(1.0, 10.0)  # Example: Random distance in meters

        # Randomly pick a predicate
        predicate = random.choice(["left", "right", "behind", "front", "above", "below"])

        # Construct a natural language query
        query = {
            "metric": f"{dist:.2f} meters",
            "semantic": semantic_label,
            "predicate": predicate
        }

        # Retrieve objects in the parent room along with room properties
        scene_objs, (room_pos, room_size) = get_scene_objects_for_room(scene_graph, parent_room)

        # Call process_query to get 3D parameters
        try:
            params = process_query(query, scene_objs)
        except Exception as e:
            print(f"Failed to process query {query}: {e}")
            continue

        # Retrieve room properties
        room_x, room_y, room_z = room_pos  # Room position
        room_width, room_depth, room_height = room_size  # Room size

        # Compute the combined PDF value at (x0, y0, z0)
        p_combined = combined_pdf(params["x0"], params["y0"], params["z0"], params)

        # Build a row for the dataset
        row_data = {
            "query": f"{query['metric']} {query['predicate']} {query['semantic']}",
            "semantic": semantic_label,
            "metric": dist,  # numeric distance for clarity
            "predicate": predicate,
            "width": width,
            "height": height,
            "depth": depth,

            # 3D spatial parameters
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

            # Room properties
            "room_x": room_x,
            "room_y": room_y,
            "room_z": room_z,
            "room_width": room_width,
            "room_height": room_height,
            "room_depth": room_depth,
            "parent_room": parent_room,

            # Computed probability
            "P_combined": p_combined
        }
        rows.append(row_data)

    # Convert rows to a Pandas DataFrame
    df = pd.DataFrame(rows)
    return df
