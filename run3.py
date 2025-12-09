from src.loader import load_dataset, load_test_images

# Load training data
train_paths, train_labels = load_dataset()

print(f"Classes found in training data: {set(train_labels)}")

print("First 5 training image paths and labels:")
for i in range(5):
    print(f" - {train_paths[i]}: {train_labels[i]}")

# Load testing data
test_paths = load_test_images()

print("First 5 test image paths:")
for path in test_paths[:5]:
    print(f" - {path}")