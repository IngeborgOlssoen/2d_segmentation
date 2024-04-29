import SimpleITK as sitk
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_nrrd_file(file_path):
    """Load an NRRD file and return a NumPy array."""
    itkimage = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(itkimage)  # Convert to NumPy array
    #print(f"Loaded {file_path}: shape {array.shape}")
    return array

def select_slices(image_array, num_slices=150):
    """Select a fixed number of slices from a 3D array evenly."""
    if image_array.shape[0] < num_slices:
        raise ValueError("Not enough slices in the volume to select 150 slices.")
    indices = np.linspace(0, image_array.shape[0] - 1, num_slices, dtype=int)
    selected_slices = image_array[indices]
    #print(f"Selected {num_slices} slices: resulting shape {selected_slices.shape}")
    return selected_slices

def process_all_files(base_path):
    categories = ['Normal', 'Diseased']
    types = ['CTCA', 'Annotations']
    data = []
    labels = []  # Labels for stratification
    for category in categories:
        for i in range(1, 21):
            image_path = os.path.join(base_path, category, types[0], f"{category}_{i}.nrrd")
            mask_path = os.path.join(base_path, category, types[1], f"{category}_{i}.nrrd")
            if os.path.exists(image_path) and os.path.exists(mask_path):
                image_volume = load_nrrd_file(image_path)
                mask_volume = load_nrrd_file(mask_path)
                selected_image_slices = select_slices(image_volume)
                selected_mask_slices = select_slices(mask_volume)
                for slice_index in range(selected_image_slices.shape[0]):
                    data.append({
                        "img": selected_image_slices[slice_index],
                        "seg": selected_mask_slices[slice_index]
                    })
                labels.extend([category] * selected_image_slices.shape[0])
                #print(f"Processed {image_path} and {mask_path}: added {len(selected_image_slices)} slices.")

    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels) #########prøve å endre 42
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels)

    #print("Data Split: Train {}, Val {}, Test {}".format(len(train_data), len(val_data), len(test_data)))
    return train_data, val_data, test_data

# Usage
base_path = "/datasets/tdt4265/mic/asoca"
train_data, val_data, test_data = process_all_files(base_path)
#print("Training Data Sample:", train_data[:2])
#print("Validation Data Sample:", val_data[:1])
#print("Testing Data Sample:", test_data[:1])
