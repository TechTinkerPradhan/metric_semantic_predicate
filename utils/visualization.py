import plotly.graph_objects as go
from metric_semantic_predicate.utils.process_query import combined_pdf
from metric_semantic_predicate.utils.data_preparation import get_scene_objects_for_room
import numpy as np


def create_box_mesh(center, size, color="lightblue", opacity=0.5, name="Box"):
    x_center, y_center, z_center = center
    w, d, h = size

    # 8 vertices
    x_coords = [
        x_center - w/2, x_center - w/2, x_center + w/2, x_center + w/2,
        x_center - w/2, x_center - w/2, x_center + w/2, x_center + w/2
    ]
    y_coords = [
        y_center - d/2, y_center + d/2, y_center + d/2, y_center - d/2,
        y_center - d/2, y_center + d/2, y_center + d/2, y_center - d/2
    ]
    z_coords = [
        z_center - h/2, z_center - h/2, z_center - h/2, z_center - h/2,
        z_center + h/2, z_center + h/2, z_center + h/2, z_center + h/2
    ]

    
    i = [0,0,4,4,0,0,1,1,2,2,3,3]
    j = [1,2,5,6,1,5,2,6,3,7,0,4]
    k = [2,3,6,7,5,4,6,5,7,6,4,7]

    mesh = go.Mesh3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        hovertext=name
    )
    return mesh


def visualize_scene_objects(scene_objects, highlight_names=None):
    """
    visualize_scene_objects(scene_objs, highlight_names=['vase'])
    - highlight_names: list of objects to color in red, rest in lightblue
    """
    if highlight_names is None:
        highlight_names = []

    highlight_set = {n.lower() for n in highlight_names}

    traces = []
    for obj in scene_objects:
        # If object's name is in highlight set, color = 'red'
        if obj.name.lower() in highlight_set:
            color = 'red'
            opacity = 0.8
        else:
            color = 'lightblue'
            opacity = 0.5

        mesh = create_box_mesh(
            center=obj.position,
            size=obj.size,
            color=color,
            opacity=opacity,
            name=obj.name
        )
        traces.append(mesh)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Scene Visualization",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    fig.show()

def plot_combined_pdf_in_room(params, query, scene_objects, room_position, room_size, resolution=30):

    # Unpack the room position & size
    rx, ry, rz = room_position
    rwidth, rdepth, rheight = room_size

    # Define axis limits based on room center & size
    x_limits = (rx - rwidth/2,  rx + rwidth/2)
    y_limits = (ry - rdepth/2, ry + rdepth/2)
    z_limits = (rz - rheight/2, rz + rheight/2)

    # Create a grid of (X, Y, Z) in the specified bounding box
    x_values = np.linspace(x_limits[0], x_limits[1], resolution)
    y_values = np.linspace(y_limits[0], y_limits[1], resolution)
    z_values = np.linspace(z_limits[0], z_limits[1], resolution)

    X, Y, Z = np.meshgrid(x_values, y_values, z_values, indexing='xy')

    # Compute the combined PDF over this grid
    P = combined_pdf(X, Y, Z, params)  # calls your combined_pdf function

    # For visualization, you might normalize so the total sum = 1
    total = np.sum(P)
    if total > 1e-12:
        P /= total

    # Create a figure
    fig = go.Figure()

    # Add the isosurface for the combined PDF
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=P.flatten(),
        isomin=P.max() * 0.1,   
        isomax=P.max(),
        surface_count=3,       # number of isosurfaces
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6,
        name='Combined PDF'
    ))

    queried_obj_name = query.get('semantic', '').lower()  # e.g. "vase"

    # Add each object's bounding box
    for obj in scene_objects:
        ox, oy, oz = obj.position
        w, d, h = obj.size

        # Decide color based on whether it matches the queried object
        if obj.name.lower() == queried_obj_name:
            color = 'red'
            opacity = 0.7
        else:
            color = 'lightblue'
            opacity = 0.5

        # 8 corners for the object bounding box
        mesh = go.Mesh3d(
            x=[
                ox - w/2, ox - w/2, ox + w/2, ox + w/2,
                ox - w/2, ox - w/2, ox + w/2, ox + w/2
            ],
            y=[
                oy - d/2, oy + d/2, oy + d/2, oy - d/2,
                oy - d/2, oy + d/2, oy + d/2, oy - d/2
            ],
            z=[
                oz,      oz,      oz,      oz,
                oz + h,  oz + h,  oz + h,  oz + h
            ],
            i=[0, 0, 0, 4, 4, 4, 2, 2, 2, 6, 6, 6],
            j=[1, 2, 3, 5, 6, 7, 3, 0, 1, 7, 4, 5],
            k=[2, 3, 1, 6, 7, 5, 0, 1, 3, 4, 5, 7],
            opacity=opacity,
            color=color,
            name=obj.name,
            hovertext=obj.name,
            showscale=False
        )
        fig.add_trace(mesh)

    # Final layout adjustments
    # Show the bounding box in the ranges of the room
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_limits, title='X'),
            yaxis=dict(range=y_limits, title='Y'),
            zaxis=dict(range=z_limits, title='Z'),
            aspectmode='data'
        ),
        title=(
            "Combined PDF Isosurface within Room<br>"
            f"Semantic: '{query.get('semantic','')}', "
            f"Metric: {query.get('metric','')}, "
            f"Predicate: '{query.get('predicate','')}'"
        )
    )

    fig.show()