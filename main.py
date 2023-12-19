import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from skimage import exposure
from sklearn.decomposition import PCA
class Image:
    def __init__(self, img, label):
        self.img = img
        self.label = label

def Load_Dataset():
    currpath="Data/Product Classification"
    Classes=os.listdir(currpath)
    Classes.sort( key=lambda x: int(x))
    Classes=list(map(lambda x:currpath+"/"+x,Classes))
    X_train,y_train,X_valid,y_valid= [], [], [], []
    trainset=[]
    validset=[]
    for Class in Classes:
        trainpart=os.listdir(Class+"/"+"Train")
        trainpart=list(map(lambda x:Class+"/"+"Train"+"/"+x,trainpart))
        for file in trainpart:
            img=cv2.imread(file,0)
            label=Class.split("/")[-1]
            trainset.append(Image(img,label))
            X_train.append(img)
            y_train.append(label)
        if "Validation"  not in os.listdir(Class):continue
        validpart=os.listdir(Class+"/"+"Validation")
        validpart=list(map(lambda x:Class+"/"+"Validation"+"/"+x,validpart))
        for file in validpart:
            img=cv2.imread(file,0)
            label=Class.split("/")[-1]
            validset.append(Image(img,label))
            X_valid.append(img)
            y_valid.append(label)
    return trainset,validset,X_train,y_train,X_valid,y_valid
def preprocessing(DataSet):
    processed_images = []
    labels = []

    for image in DataSet:
        img = image.img
        # Resize the image
        new_width, new_height = 300, 200
        resized_image = cv2.resize(img, (new_width, new_height))

        # Normalize the image
        normalized_image = resized_image / 255.0

        # Gaussian Blur
        blurred_image = cv2.GaussianBlur(normalized_image, (3, 3), 0)

        processed_images.append(blurred_image)
        labels.append(image.label)

    return np.array(processed_images), np.array(labels)
#1- features extraction
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        if(img==None): return Exception("img is None")
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            descriptors_list.append(des)
    return descriptors_list

#2 - build vocabulary to finite the descriptors
def build_vocabulary(descriptors_list, vocab_size):
    descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=vocab_size, random_state=1900)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_

#3 - build histogram based on vocabulary and descriptors
def build_histograms(vocabulary, descriptors_list):
    histograms = []
    for des in descriptors_list:
        histogram = np.zeros(len(vocabulary))
        for d in des:
            idx = np.argmin(np.linalg.norm(vocabulary - d, axis=1))
            histogram[idx] += 1
        histograms.append(histogram)
    return histograms

trainset,validset,X_train,y_train,X_valid,y_valid=Load_Dataset()
X_train_processed, y_train_processed = preprocessing(trainset)
X_valid_processed, y_valid_processed = preprocessing(validset)

X_train_desc=extract_sift_features(X_train_processed)
X_valid_desc=extract_sift_features(X_valid_processed)

vocab=build_vocabulary(X_train_desc, 128)

X_train_hist=build_histograms(vocab, X_train_desc)
X_valid_hist=build_histograms(vocab, X_valid_desc)




