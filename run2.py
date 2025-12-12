import os
import cv2
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

PATCH_SIZE = 8
STRIDE = 4
TRAIN_PATH = './training'
# Maximum number of patches to sample per image 
IMG_SUBSAMPLE = 200
# n_init for KMeans
KMEANS_RERUNS = 5
# Range of K to test
K_VALUES_TO_TEST = range(475, 525, 5) 

def extract_dense_patches(img_path):
    """
    Reads image in grayscale, extracts 8x8 patches with stride 4.
    Returns a list of flattened, normalized vectors.
    """
    # Could remove since images are greyscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None
    
    h, w = img.shape
    patches = []

    for y in range(0, h - PATCH_SIZE, STRIDE):
        for x in range(0, w - PATCH_SIZE, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            # Flatten
            vec = patch.flatten().astype(np.float32)
            
            # Mean-centring
            vec -= np.mean(vec)

            # Normalisation
            norm = np.linalg.norm(vec)
            if norm > 0: # In case patch was constant
                vec /= norm
            
            patches.append(vec)
            
    return np.array(patches)

def collect_patches_for_vocab(image_paths):
    """
    Extracts patches from images once to be used for repeated K-Means clustering.
    """
    all_patches = []
    
    for i, path in enumerate(image_paths):
        # Extract patches from image
        patches = extract_dense_patches(path)
        if patches is not None:
            # Subsampling for KMeans
            if len(patches) > IMG_SUBSAMPLE:
                # Pick IMG_SUBSAMPLE random patches per image
                indices = np.random.choice(len(patches), IMG_SUBSAMPLE, replace=False)
                all_patches.append(patches[indices])
            else: # Fewer than IMG_SUBSAMPLE patches
                all_patches.append(patches)

    # Stack patches 
    training_data = np.vstack(all_patches)
    return training_data

def train_kmeans(training_data, k):
    """
    Trains K-Means on the pre-loaded data.
    """
    print(f"  Running K-Means with k={k}")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_RERUNS)
    kmeans.fit(training_data)
    return kmeans

def get_image_histogram(img_path, kmeans):
    """
    Converts image into normalised histogram based on the visual words
    """
    patches = extract_dense_patches(img_path)

    if patches is None or len(patches) == 0:
        return np.zeros(kmeans.n_clusters)
        
    predictions = kmeans.predict(patches)
    
    # Histogram of predicted patch clusters (counts frequency of visual words)
    hist, _ = np.histogram(predictions, bins=range(kmeans.n_clusters + 1))
    
    # Normalise histogram (so histogram is invariant to image size)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist.astype(np.float32) / norm
        
    return hist

def load_dataset(directory, kmeans):
    """
    Dataset Loader
    """
    X = []
    y = []

    # Sorted since listdir produces arbitrary orders
    classes = sorted(os.listdir(directory))
    
    for label in classes:
        class_dir = os.path.join(directory, label) # Build path to subfolder
        if not os.path.isdir(class_dir):
            continue
        image_files = glob(os.path.join(class_dir, "*.jpg")) # Find all jpeg files
        
        for img_file in image_files:
            hist = get_image_histogram(img_file, kmeans)
            X.append(hist) # Histogram of features
            y.append(label) # Label (folder name)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Get list of training images
    train_image_paths = []
    for root, dirs, files in os.walk(TRAIN_PATH):
        for file in files:
            if file.endswith(".jpg"):
                train_image_paths.append(os.path.join(root, file))

    # Pre-extract patches
    patch_data = collect_patches_for_vocab(train_image_paths)

    best_acc = 0.0
    best_k = 0
    results = {}

    print(f"\nStarting Grid Search for k-values {K_VALUES_TO_TEST}")

    # Training KMeans for range of k values
    for k in K_VALUES_TO_TEST:
        print(f"Testing k value = {k}")

        # Trains KMeans
        kmeans_model = train_kmeans(patch_data, k)

        # Generate features from images
        print("  Generating Training Histograms...")
        X_train, y_train = load_dataset(TRAIN_PATH, kmeans_model)

        # Train linear ova classifiers
        clf = LinearSVC(C=1.0, max_iter=2000, multi_class='ovr')
        clf.fit(X_train, y_train)

        # Get Training accuracy
        y_pred = clf.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        
        print(f"Training accuracy: {acc*100:.2f}%")
        
        # Store accuracy of model
        results[k] = acc
        
        # Update best model
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print("\n================================================")
    print(f"Best k: {best_k}")
    print(f"Best accuracy:   {best_acc*100:.2f}%")
    print("================================================")


    # =========================================================
    # OPTIMAL K VALUE WAS 515
    # =========================================================

    TEST_PATH = './testing'
    OUTPUT_FILE = 'run2.txt'
    FINAL_K = 515 
    
    print(f"\nGenerating {OUTPUT_FILE} using k={FINAL_K}")

    # Retrain the model on the specific K value 
    final_kmeans = train_kmeans(patch_data, FINAL_K)
    
    # Generate features from images
    X_train_final, y_train_final = load_dataset(TRAIN_PATH, final_kmeans)
    
    # Train linear ova classifiers
    final_clf = LinearSVC(C=1.0, max_iter=2000, multi_class='ovr')
    final_clf.fit(X_train_final, y_train_final)
    
    # Find testing images
    test_files = glob(os.path.join(TEST_PATH, "*.jpg"))
    
    # Sort files numerically
    try:
        test_files.sort(key=lambda f: int(os.path.basename(f).split('.')[0]))
    except ValueError:
        # If files aren't numbers
        test_files.sort()

    with open(OUTPUT_FILE, 'w') as f:
        for file_path in test_files:
            # Generate histogram for test data
            hist = get_image_histogram(file_path, final_kmeans)
            
            # Predict (reshape to 1, -1 because it's a single sample)
            prediction = final_clf.predict([hist])[0]
            
            # Write to file
            filename = os.path.basename(file_path)
            f.write(f"{filename} {prediction}\n")