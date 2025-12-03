import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def loadImagePathsAndLabels(trainDir='training', testDir='testing'):
    trainImagePaths = []
    trainLabels = []
    if os.path.exists(trainDir):
        for className in sorted(os.listdir(trainDir)):
            classPath = os.path.join(trainDir, className)
            if os.path.isdir(classPath):
                for imageFile in sorted(os.listdir(classPath)):
                    if imageFile.lower().endswith(('.jpg', '.jpeg', '.png')):
                        imagePath = os.path.join(classPath, imageFile)
                        trainImagePaths.append(imagePath)
                        trainLabels.append(className)
    else:
        raise FileNotFoundError(f"Training directory '{trainDir}' not found")
    testImagePaths = []
    if os.path.exists(testDir):
        for imageFile in sorted(os.listdir(testDir), key=lambda x: int(os.path.splitext(x)[0])):
            if imageFile.lower().endswith(('.jpg', '.jpeg', '.png')):
                imagePath = os.path.join(testDir, imageFile)
                testImagePaths.append(imagePath)
    else:
        raise FileNotFoundError(f"Test directory '{testDir}' not found")
    return trainImagePaths, trainLabels, testImagePaths

def tinyImage(path, size=16):
    image = Image.open(path).convert('L')
    imageArray = np.array(image, dtype=np.float32)
    height, width = imageArray.shape
    side = min(height, width)
    startHeight = (height - side) // 2
    startWidth = (width - side) // 2
    cropped = imageArray[startHeight:startHeight + side, startWidth:startWidth + side]
    resizedImage = Image.fromarray(cropped.astype(np.uint8))
    resizedImage = resizedImage.resize((size, size), Image.Resampling.LANCZOS)
    resizedArray = np.array(resizedImage, dtype=np.float32)
    featureVector = resizedArray.flatten()
    featureVector = featureVector - np.mean(featureVector)
    norm = np.linalg.norm(featureVector)
    if norm > 0:
        featureVector = featureVector / norm
    return featureVector

def extractFeatures(imagePaths, size=16):
    features = []
    for path in imagePaths:
        features.append(tinyImage(path, size))
    return np.array(features)

def writePredictions(testImagePaths, predictions, outputFile='run1.txt'):
    with open(outputFile, 'w') as file:
        for path, prediction in zip(testImagePaths, predictions):
            name = os.path.basename(path)
            file.write(f"{name} {prediction}\n")

def main():
    trainImagePaths, trainLabels, testImagePaths = loadImagePathsAndLabels()
    trainFeatures = extractFeatures(trainImagePaths, size=16)
    testFeatures = extractFeatures(testImagePaths, size=16)
    
    k_values = [1, 3, 5, 7, 9, 11, 15, 16, 20, 25, 30]
    metrics = ['euclidean', 'cosine']
    weights_options = ['uniform', 'distance']
    
    best_k = 7
    best_metric = 'cosine'
    best_weights = 'distance'
    best_score = 0
    
    for metric in metrics:
        for weights in weights_options:
            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
                scores = cross_val_score(knn, trainFeatures, trainLabels, cv=5, scoring='accuracy')
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k
                    best_metric = metric
                    best_weights = weights
    
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=best_weights)
    knn.fit(trainFeatures, trainLabels)
    testPredictions = knn.predict(testFeatures)
    writePredictions(testImagePaths, testPredictions, outputFile='run1.txt')

if __name__ == "__main__":
    main()
