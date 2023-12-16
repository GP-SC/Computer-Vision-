import numpy as np
import cv2
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from skimage import exposure

class Image:
    def __init__(self, img, label):
        self.img = img
        self.label = label
def Load_Dataset():
    currpath="/kaggle/input/computer-vision-class/Data/Product Classification"
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
        if len(img.shape) == 2:  # Check if the image is grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

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
trainset,validset,X_train,y_train,X_valid,y_valid=Load_Dataset()
X_train_processed, y_train_processed = preprocessing(trainset)
X_valid_processed, y_valid_processed = preprocessing(validset)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_processed)
y_valid_encoded = label_encoder.transform(y_valid_processed)


input_shape = X_train_processed[0].shape + (1,)


X_train_reshaped = X_train_processed.reshape(-1, 300, 200, 1)
X_test_reshaped = X_valid_processed.reshape(-1, 300, 200, 1)

