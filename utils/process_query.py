""" Process input query and generate parameters for PDF logic."""
import re
import numpy as np

def process_query(query, scene_objects, sigma_factor=0.5, sigma_m_factor=0.3, kappa_factor=10.0):
    """
    Processes a query (with keys: 'metric', 'semantic', 'predicate')
    and generates params for your PDF logic.

    e.g. query = {
      'metric': '2 meters',
      'semantic': 'vase',
      'predicate': 'left of'
    }
    """
    

    # 1) Parse numeric distance
    metric_str = query['metric']  # e.g. "2 meters"
    match = re.search(r'(\d+(\.\d+)?)', metric_str)
    if match:
        desired_distance = float(match.group(1))
    else:
        desired_distance = 0.0

    # Find object in scene_objects
    semantic_label = query['semantic']
    obj = next((o for o in scene_objects if o.name.lower() == semantic_label.lower()), None)
    if obj is None:
        raise ValueError(f"Object '{semantic_label}' not found among scene objects.")

    # Extract object position & size
    mu_x, mu_y, mu_z = obj.position
    width, depth, height = obj.size

    sigma_s = sigma_factor * max(width, depth, height)
    sigma_m = sigma_m_factor * max(width, depth, height)
    kappa = kappa_factor / max(width, depth, height)

    x0, y0, z0 = mu_x, mu_y, mu_z
    d0 = desired_distance

    # Convert predicate => angles 
    predicate_directions = {
        "left":   (np.pi,     np.pi / 2),
        "right":  (0.0,       np.pi / 2),
        "front":  (np.pi / 2, np.pi / 2),
        "behind":    (3*np.pi/2, np.pi / 2),
        "above":     (0.0,       0.0),
        "below":     (0.0,       np.pi)
    }
    predicate = query['predicate']
    if predicate not in predicate_directions:
        raise ValueError(f"Unknown predicate: '{predicate}'. Must be one of: {list(predicate_directions.keys())}")

    theta0, phi0 = predicate_directions[predicate]

    # Build params
    params = {
        'mu_x': mu_x,  'mu_y': mu_y,  'mu_z': mu_z,
        'sigma_s': sigma_s,
        'x0': x0, 'y0': y0, 'z0': z0,
        'd0': d0,
        'sigma_m': sigma_m,
        'theta0': theta0,
        'phi0': phi0,
        'kappa': kappa
    }
    return params
