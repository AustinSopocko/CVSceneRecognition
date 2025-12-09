import os
import glob

def load_dataset(base_path = './data/training'):
    """        
    Args:
        base_path (str): Path to the training data folder (e.g., './data/training').

    Returns:
        image_paths (list): List of full file paths to the training images.
        labels (list): List of string labels corresponding to each image path.
    """
    image_paths = []
    labels = []
    
    # We iterate through each folder in the training directory.
    classes = sorted(os.listdir(base_path))

    for class_name in classes:
        class_dir = os.path.join(base_path, class_name)
        
        # Ensure we are only looking at directories (ignore hidden files)
        if not os.path.isdir(class_dir):
            continue
        # Get all JPEG images in the class directory
        search_path = os.path.join(class_dir, '*.jpg')
        class_files = glob.glob(search_path)
        
        print(f"Loaded {len(class_files)} images for class: {class_name}")
        
        for file_path in class_files:
            image_paths.append(file_path)
            labels.append(class_name)
            
    return image_paths, labels

def load_test_images(test_path = './data/testing'):
    """    
    Args:
        test_path (str): Path to the testing data folder (e.g., './data/testing').

    Returns:
        test_paths (list): Sorted list of full file paths to the test images.
    """
    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"Testing directory not found at: {test_path}")

    # Retrieve all .jpg images
    search_path = os.path.join(test_path, '*.jpg')
    test_paths = glob.glob(search_path)
    
    test_paths.sort()
    
    print(f"Total testing images found: {len(test_paths)}")
    
    return test_paths