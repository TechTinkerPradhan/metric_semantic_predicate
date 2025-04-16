# === Object Extraction Helpers ===

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

def get_object_sizes(scene_graph):
    """
    Returns a dictionary mapping each object's class name (e.g. "chair", "microwave")
    to its (bounding_box_size_x, bounding_box_size_y, bounding_box_size_z).

    Example output:
      {
        "chair": (0.46, 0.54, 0.92),
        "microwave": (0.17, 0.15, 0.23),
        ...
      }

    If multiple objects share the same class name, this overwrites by the last one encountered.
    Adjust as needed if you have duplicates or want a list of all sizes per class.
    """
    obj_sizes = {}

    for obj_id, obj_data in scene_graph["object"].items():
        name = obj_data["class_"]  # e.g. "chair", "microwave", etc.
        
        # Assuming your scene graph dictionary keys for bounding box size are exactly:
        # "bounding box size X", "bounding box size Y", and "bounding box size Z"
        # If the keys differ, update accordingly.
        size_x = obj_data["bounding box size X"]
        size_y = obj_data["bounding box size Y"]
        size_z = obj_data["bounding box size Z"]
        
        obj_sizes[name] = (size_x, size_y, size_z)

    return obj_sizes