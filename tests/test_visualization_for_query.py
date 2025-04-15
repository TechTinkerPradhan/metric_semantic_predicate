import re
import numpy as np
from metric_semantic_predicate.utils.data_preparation import get_scene_objects_for_room
from metric_semantic_predicate.utils.visualization import visualize_scene_objects
from metric_semantic_predicate.utils.llama_helpers import run_llama, extract_structure
from metric_semantic_predicate.data.scene_graph_loader import scene_graph
# Example user query
user_query = "go 2 meters left of bed"

# prompt for Llama
full_query = (
    f"Given the following natural language query: '{user_query}' "
    "decompose it into a structured query with the following components: "
    "1. Metric: Extract any numerical or distance-related information, "
    "2. Semantic: Extract the object(s) involved in the query, "
    "3. Predicate: Extract the spatial or logical relationship between the objects. "
    "Output the structured query in the exact format below and nothing else: "
    "Structured Query: Metric: [Extracted numerical or distance-related information], "
    "Semantic: [Extracted object(s)], "
    "Predicate: [Extracted spatial or logical relationship]. "
    "Example Input: 'go 2 meters left of the refrigerator'. "
    "Example Output: 'Metric: 2 meters, Semantic: refrigerator, Predicate: left of'. "
    "Now, generate the structured query."
)

# Run Llama
structured_response = run_llama(full_query)
print("Structured Response from Llama:")
print(structured_response)

# Extract metric, semantic, predicate
metric, semantic, predicate = extract_structure(structured_response)
print("\nParsed Values:")
print("  Metric:", metric)
print("  Semantic:", semantic)
print("  Predicate:", predicate)

# room we're checking
user_input_room_number = 5

# Get all objects in the user's chosen room
scene_objs, (room_pos, room_size) = get_scene_objects_for_room(scene_graph, user_input_room_number)

# Display which objects are in this room
print(f"\nRoom {user_input_room_number} contains these objects:")
for o in scene_objs:
    print("  ", o)

# From Llama's semantic, parse the first word (e.g. "vase")
object_name = None
if semantic:
    match = re.match(r"^(\w+)", semantic)
    if match:
        object_name = match.group(1)

# Check if the user-specified object_name is among them
found_object = None
if object_name:
    for o in scene_objs:
        if o.name.lower() == object_name.lower():
            found_object = o
            break

if not found_object:
    print(f"\nNo objects named '{object_name}' in room {user_input_room_number}.")
else:
    print(f"\nObject '{object_name}' found in room {user_input_room_number}: {found_object}")
    # Visualize all objects in the room, highlighting the found object in red
    visualize_scene_objects(scene_objs, highlight_names=[object_name])
