import numpy as np
import pandas as pd
from metric_semantic_predicate.utils.pdf_generator import combined_pdf
import argparse
import os


def generate_synthetic_human_data(df_original, num_rows=1000, human_altered=600, random_state=42):
    """
    Generate a synthetic 'human' dataset with the same columns as df_original,
    but ONLY add random noise to:
      - sigma_s
      - theta0
      - phi0
      - kappa
      - P_combined (computed from updated noisy parameters)

    All other columns remain unchanged from the baseline row.

    Parameters
    ----------
    df_original : pd.DataFrame
        The original dataset with all required columns.
    num_rows : int
        Total number of rows in the synthetic human dataset to generate.
    human_altered : int
        Number of rows (from the end) to which we apply bigger noise.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    df_human : pd.DataFrame
        A new DataFrame with 'num_rows' rows and the same columns as df_original,
        but with random perturbations ONLY in sigma_s, theta0, phi0, kappa, and P_combined.
    """

    # Required columns
    required_cols = [
        "query", "semantic", "metric", "predicate",
        "width", "height", "depth",
        "mu_x", "mu_y", "mu_z", "sigma_s",
        "x0", "y0", "z0", "d0", "sigma_m",
        "theta0", "phi0", "kappa",
        "room_x", "room_y", "room_z", "room_width", "room_height", "room_depth", "parent_room",
        "P_combined"
    ]
    for col in required_cols:
        if col not in df_original.columns:
            raise ValueError(f"Column '{col}' not found in df_original.")

    # Convert df_original to a new index so sampling is easier
    df_original = df_original.reset_index(drop=True)
    n_original = len(df_original)

    # Prepare a container for new rows
    synthetic_rows = []
    rng = np.random.default_rng(random_state)

    # For each of the num_rows we want to create
    for i in range(num_rows):
        # Pick a random baseline row
        baseline_idx = rng.integers(0, n_original)
        baseline = df_original.iloc[baseline_idx]

        # Copy over the baseline row
        row_dict = {col: baseline[col] for col in required_cols}

        # Check if this row falls in the 'human_altered' range
        is_human = (i >= (num_rows - human_altered))

        # 1) Add noise to sigma_s
        noise_s_sigma = 0.1 if not is_human else 0.2
        row_dict["sigma_s"] += rng.normal(0, noise_s_sigma)
        row_dict["sigma_s"] = max(row_dict["sigma_s"], 0.01)  # Ensure positive

        # 2) Add noise to theta0, phi0
        noise_theta = 0.4 if not is_human else 0.8
        noise_phi = 0.3 if not is_human else 0.6

        row_dict["theta0"] += rng.normal(0, noise_theta)
        row_dict["phi0"] += rng.normal(0, noise_phi)

        # Clip phi0 to [0, pi]
        row_dict["phi0"] = np.clip(row_dict["phi0"], 0, np.pi)

        # 3) Add noise to kappa
        noise_kappa = 5.0 if not is_human else 10.0
        row_dict["kappa"] += rng.normal(0, noise_kappa)
        row_dict["kappa"] = max(row_dict["kappa"], 0.01)  # Ensure positive

        # 4) Recompute P_combined using the modified values
        params = {
            "mu_x": row_dict["mu_x"], "mu_y": row_dict["mu_y"], "mu_z": row_dict["mu_z"], "sigma_s": row_dict["sigma_s"],
            "x0": row_dict["x0"], "y0": row_dict["y0"], "z0": row_dict["z0"], "d0": row_dict["d0"], "sigma_m": row_dict["sigma_m"],
            "theta0": row_dict["theta0"], "phi0": row_dict["phi0"], "kappa": row_dict["kappa"]
        }
        row_dict["P_combined"] = combined_pdf(row_dict["x0"], row_dict["y0"], row_dict["z0"], params)

        # Add to synthetic list
        synthetic_rows.append(row_dict)

    # Convert list to DataFrame
    df_human = pd.DataFrame(synthetic_rows, columns=required_cols)
    return df_human

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate human-altered dataset with noise.")
    parser.add_argument(
        "--file",
        type=str,
        default="3DSceneGraph_Beechwood_dataset.csv",
        help="Path to the original dataset CSV (default: 3DSceneGraph_Beechwood_dataset.csv in /data)"
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=1000,
        help="Number of synthetic rows to generate (default: 1000)"
    )
    parser.add_argument(
        "--human_altered",
        type=int,
        default=None,
        help="Number of rows to apply stronger human-like noise (default: 50% of num_rows)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Resolve input file path
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    input_path = args.file if os.path.isabs(args.file) else os.path.join(data_dir, args.file)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find the input file at: {input_path}")

    # Load original dataset
    df_original = pd.read_csv(input_path)
    print(f"Loaded: {input_path} with {len(df_original)} rows")

    # Default human_altered to 50% of num_rows if not specified
    human_altered = args.human_altered or int(0.5 * args.num_rows)

    # Generate simulated human dataset
    df_human = generate_synthetic_human_data(
        df_original,
        num_rows=args.num_rows,
        human_altered=human_altered,
        random_state=args.random_state
    )

    # Save to data/ as <original_name>_human.csv
    file_base = os.path.splitext(os.path.basename(input_path))[0]
    output_file = f"{file_base}_human.csv"
    output_path = os.path.join(data_dir, output_file)

    df_human.to_csv(output_path, index=False)
    print(f"Saved simulated human dataset to: {output_path}")