import glob
from PIL import Image
import numpy
import PIL
import os
import matplotlib.pyplot as plt
import cv2
import math
import codecs
from sklearn.decomposition import PCA
import JsonEncoder as json

countOfTestImages = 100
countOfTrainImages = 2000
numOfPrincipalComponents = 10

firstSetTrainImagesDirectories = ["Data\\MNIST\\Train\\3", "Data\\MNIST\\Train\\6"]
firstSetTestImagesDirectories = ["Data\\MNIST\\Test\\3", "Data\\MNIST\\Test\\6"]
first_set_json_data_file_path = "Data\\MNIST\\mnist_pca_" + str(numOfPrincipalComponents) +"Components_3_6.json"

secondSetTrainImagesDirectories = ["Data\\MNIST\\Train\\1", "Data\\MNIST\\Train\\6"]
secondSetTestImagesDirectories = ["Data\\MNIST\\Test\\1", "Data\\MNIST\\Test\\6"]
second_set_json_data_file_path = "Data\\MNIST\\mnist_pca_" + str(numOfPrincipalComponents) +"Components_1_6.json"

def apply_pca(imageFiles, numImages):

    normalizationFactor = 250

    pcaComponents = PCA(n_components=numOfPrincipalComponents)
    projected = pcaComponents.fit_transform(imageFiles)
    #prinicipleComponents = pcaComponents.components_

    imageVectors = projected.reshape(numImages, numOfPrincipalComponents)/normalizationFactor

    return imageVectors

def extract_features_labels(trainImageDirectories, testImageDirectories):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for label in [0, 1]:

        #Combine train and test data for single class of images
        imageFilePaths = [f for f in glob.glob(trainImageDirectories[label] + "**/*.jpg")]
        imageFilePaths = imageFilePaths + [f for f in glob.glob(testImageDirectories[label] + "**/*.jpg")]

        # Assuming all images are the same size, get dimensions of first image
        width, height = Image.open(imageFilePaths[0]).size
        numberOfImages = len(imageFilePaths)        

        #Read all images into flat arrays of features
        imageFiles = numpy.array([numpy.array(Image.open(im)).flatten()
                                for im in imageFilePaths], 'f')
        
        #Reduce number of features by applying PCA
        imageVectors = apply_pca(imageFiles, numberOfImages);

        train_labels = train_labels + [label]*countOfTrainImages
        test_labels = test_labels + [label]*countOfTestImages
        
        train_features.extend(imageVectors[0:countOfTrainImages])
        test_features.extend(imageVectors[countOfTrainImages:])

    return train_features, test_features, train_labels, test_labels;

def dump_to_json(trainingFeatures, trainingLabels, validationFeatures, validationLabels, json_path):
    feature_label_data = {
        "TrainingData": {
            "Features": numpy.array(trainingFeatures).tolist(),
            "Labels": trainingLabels
        },
        "ValidationData":{
            "Features": numpy.array(validationFeatures).tolist(),
            "Labels": validationLabels
        }
    }

    json.dump(feature_label_data, json_path, 4)


trainFeat, testFeat, trainLab, testLab = extract_features_labels(firstSetTrainImagesDirectories, firstSetTestImagesDirectories)

dump_to_json(trainFeat, trainLab, testFeat, testLab, first_set_json_data_file_path)