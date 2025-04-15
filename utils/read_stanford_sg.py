import numpy as np
file_path='/home/artemis/project/scene_graph/data/verified_graph/output2.npy'

data = np.load(file_path, allow_pickle=True).item()
print(f"Data Type: {type(data)}")

# If the data is a dictionary or a list of dictionaries, we can access its contents
if isinstance(data, dict):
    print("Keys:", data.keys())
    sample_key = list(data.keys())[2]
    print(f"Sample Data for Key '{sample_key}':", data[sample_key])
elif isinstance(data, list):
    print("Length of List:", len(data))
    if len(data) > 0:
        sample_data = data[0]
        print("Sample Data:", sample_data)
        if isinstance(sample_data, dict):
            print("Sample Data Keys:", sample_data.keys())
else:
    print("Data is not a list or dictionary, here is the raw content:", data)