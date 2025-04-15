
import numpy as np

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