# scripts/generate_function_dataset.py

import argparse
import os
import pandas as pd

from metric_semantic_predicate.utils.scene_graph_loader import load_scene_graphs_by_name
from metric_semantic_predicate.dataset.dataset_generator import generate_3d_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D dataset from scene graphs.")
    parser.add_argument(
        "--houses",
        nargs="+",
        default=["Beechwood"],
        help="List of house names to generate datasets for (default: Beechwood)"
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=1000,
        help="Number of spatial queries to generate per house (default: 1000)"
    )
    args = parser.parse_args()

    scene_graphs = load_scene_graphs_by_name(args.houses)

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(save_dir, exist_ok=True)

    for house, scene_graph in scene_graphs.items():
        print(f"📦 Generating dataset for {house}...")
        df_dataset = generate_3d_dataset(scene_graph, num_queries=args.num_queries)
        path = os.path.join(save_dir, f"{house}_dataset.csv")
        df_dataset.to_csv(path, index=False)
        print(f"✅ Saved: {path}")
