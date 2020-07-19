import glob
from PIL import Image
import numpy
import PIL
import os
import matplotlib.pyplot as plt
import cv2
import math
import codecs
import json
from sklearn.decomposition import PCA

def extract_features_labels(imageDirectories):
    label = 0
    features = []
    labels = []

    for imagesDirectory in imageDirectories:
        imageFilePaths = [f for f in glob.glob(imagesDirectory + "**/*.jpg")]

        # Assuming all images are the same size, get dimensions of first image
        width, height = Image.open(imageFilePaths[0]).size
        numberOfImages = len(imageFilePaths)

        # Create a numpy array of floats to store the average
        arr = numpy.zeros((height, width), numpy.float)

        # Build up average pixel intensities, casting each image as an array of floats
        for im in imageFilePaths:
            imgFile = Image.open(im)
            imageMat = numpy.array(imgFile, dtype=numpy.float)
            arr = arr+imageMat/numberOfImages

        imageFiles = numpy.array([numpy.array(Image.open(im)).flatten()
                                for im in imageFilePaths], 'f')

        pca10Components = PCA(n_components=10)
        projected10 = pca10Components.fit_transform(imageFiles)
        prinicipleComponents = pca10Components.components_

        # Round values in array and cast as 8-bit integer
        arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)

        imageVectors = projected10.reshape(numberOfImages, 10)/200
        features = features + imageVectors.tolist()

        labels = labels + [label]*numberOfImages
        label = label + 1

    return numpy.round(features, 2).tolist(), labels;

trainImagesDirectories = ["Data\\MNIST\\Train\\3", "Data\\MNIST\\Train\\6"]
testImagesDirectories = ["Data\\MNIST\\Test\\3", "Data\\MNIST\\Test\\6"]
json_data_file_path = "D:\\GDrive\\MSc\\Dissertation\\git\\QuantumExperiments\\QDK\\Data\\MNIST\\mnist_pca.json"

trainingFeatures, trainingLabels = extract_features_labels(trainImagesDirectories)
validationFeatures, validationLabels = extract_features_labels(testImagesDirectories)  

x = {
    "TrainingData": {
        "Features": trainingFeatures,
        "Labels": trainingLabels
    },
    "ValidationData":{
        "Features": validationFeatures,
        "Labels": validationLabels
    }
}

json.dump(x, codecs.open(json_data_file_path, 'w', encoding='utf-8'),
          separators=(',', ':'), sort_keys=True, indent=4)
