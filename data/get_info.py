import os
import json
from collections import defaultdict

# Base directory
base_dir = 'data'
categories = ['awake','sleepy']
splits = ['train', 'val', 'test']

# Function to count images in each split/category
def count_images():
    image_counts = defaultdict(dict)
    for split in splits:
        for category in categories:
            folder_path = os.path.join(base_dir, split, category)
            num_images = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            image_counts[split][category] = num_images
    return image_counts

# Function to generate directory tree without file names
def generate_tree(startpath):
    tree_structure = []
    for dirpath, dirnames, _ in os.walk(startpath):  # Ignore filenames
        level = dirpath.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_structure.append(f"{indent}{os.path.basename(dirpath)}/")
    return '\n'.join(tree_structure)


# Function to generate schema
def generate_schema():
    schema = {
        "subject_id": "Unique identifier for each subject (37 subjects)",
        "image_id": "Unique identifier for each image",
        "attributes": {
            "eye_state": "0 for Sleepy, 1 for Awake",
            "gender": "0 for male, 1 for female",
            "glasses": "0 for no, 1 for yes",
            "reflections": "0 for none, 1 for small, 2 for big",
            "lighting_conditions": "0 for bad, 1 for good",
            "sensor_id": {
                "01": "RealSense (640x480)",
                "02": "IDS (1280x1024)",
                "03": "Aptina (752x480)"
            }
        }
    }
    return schema

# Execute and print results
if __name__ == "__main__":
    # Get image counts
    counts = count_images()
    print("Image counts per split and category:")
    print(json.dumps(counts, indent=4))

    # Get directory tree
    tree = generate_tree(base_dir)
    print("\nDirectory tree:")
    print(tree)

    # Get schema details
    schema = generate_schema()
    print("\nDataset schema:")
    print(json.dumps(schema, indent=4))
