import cv2
import numpy as np
import math
from sklearn.metrics import pairwise_distances_argmin

def extract_dense_sift(img, step_size=8, feature_scale=8):
    """
    Extracts SIFT descriptors on a dense grid (rather than at interesting keypoints).
    
    Args:
        img (numpy.ndarray): Grayscale image.
        step_size (int): Distance in pixels between each feature.
                         Smaller = more features (slower but more detailed).
        feature_scale (int): Size of the SIFT patch (diameter).

    Returns:
        keypoints (list): List of cv2.KeyPoint objects (contains x, y coordinates).
        descriptors (numpy.ndarray): N x 128 matrix of SIFT descriptors.
    """
    # 1. Create SIFT object
    sift = cv2.SIFT_create()

    # We loop over the image height and width with the given step size
    h, w = img.shape
    keypoints = []
    
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            # cv2.KeyPoint(x, y, size)
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(feature_scale)))

    # 3. Compute descriptors at those specific grid points
    # descriptors will be None if the image is empty or no keypoints are valid
    keypoints, descriptors = sift.compute(img, keypoints)
    
    if descriptors is None:
        descriptors = np.zeros((0, 128), dtype=np.float32)

    return keypoints, descriptors

def build_spatial_pyramid(image_path, codebook, levels=2):
    """
    Builds a Spatial Pyramid Matching (SPM) histogram for a single image.
    
    Args:
        image_path (str): Path to the image file.
        codebook (sklearn.cluster.KMeans or similar): Trained KMeans model.
                                                      Must have .cluster_centers_.
        levels (int): Number of pyramid levels.
                      Level 0 = global histogram (1x1)
                      Level 1 = 2x2 grid
                      Level 2 = 4x4 grid
                      
    Returns:
        final_histogram (numpy.ndarray): Concatenated, normalized 1D feature vector.
    """
    # 1. Load image (already in grayscale)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    h, w = img.shape

    # 2. Extract Dense SIFT features
    keypoints, descriptors = extract_dense_sift(img, step_size=8)
    
    if len(keypoints) == 0:
        # Edge case: Image is blank or too small. Return zero vector.
        # Size = vocabulary_size * (1 + 4 + 16 + ... 4^L)
        vocab_size = codebook.cluster_centers_.shape[0]
        total_bins = vocab_size * sum([4**l for l in range(levels + 1)])
        return np.zeros(total_bins, dtype=np.float32)

    # 3. Quantize Descriptors to Visual Words
    # We find the index of the closest cluster center for each descriptor.
    # Use sklearn.metrics.pairwise_distances_argmin to assign nearest center
    visual_words = pairwise_distances_argmin(descriptors, codebook.cluster_centers_)
    
    # 4. Build Histograms for each Level
    pyramid_histograms = []
    vocab_size = len(codebook.cluster_centers_)
    
    for level in range(levels + 1):
        # Number of cells along one dimension: 2^level (1, 2, 4...)
        num_cells = 2 ** level
        
        # Calculate cell size
        cell_h = np.ceil(h / num_cells)
        cell_w = np.ceil(w / num_cells)
        
        # Iterate through the grid cells (i = row, j = col)
        for i in range(num_cells):
            for j in range(num_cells):
                
                # Define cell boundaries
                y_min, y_max = i * cell_h, (i + 1) * cell_h
                x_min, x_max = j * cell_w, (j + 1) * cell_w
                
                # Filter features that fall inside this cell
                # We check the x, y coordinates of every keypoint
                # (This can be optimized with numpy masking, but loop is clearer for learning)
                cell_word_indices = []
                
                for k, kp in enumerate(keypoints):
                    if (x_min <= kp.pt[0] < x_max) and (y_min <= kp.pt[1] < y_max):
                        cell_word_indices.append(visual_words[k])
                
                # Compute histogram for this cell
                # bins range from 0 to vocab_size
                hist, _ = np.histogram(cell_word_indices, bins=vocab_size, range=(0, vocab_size))
                
                # Normalize the local histogram (L1 normalization)
                # This ensures images with more texture don't dominate just because they have more points
                norm = np.linalg.norm(hist, ord=1)
                if norm > 0:
                    hist = hist / norm
                
                pyramid_histograms.append(hist)

    # 5. Concatenate all histograms into one long vector
    final_vector = np.concatenate(pyramid_histograms)
    
    # 6. Global Normalization (L1)
    # Important for the Hellinger Kernel later
    final_norm = np.linalg.norm(final_vector, ord=1)
    if final_norm > 0:
        final_vector = final_vector / final_norm

    return final_vector