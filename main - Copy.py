import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
import sys
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
    mx=(0,0)
    for Class in Classes:
        trainpart=os.listdir(Class+"/"+"Train")
        trainpart=list(map(lambda x:Class+"/"+"Train"+"/"+x,trainpart))
        for file in trainpart:
            img=cv2.imread(file,1)
            label=Class.split("/")[-1]
            mx=max(mx,img.shape)
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
        img=cv2.resize(img,(500,500))
        # Normalize the image
        normalized_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # Gaussian Blur
        blurred_image = cv2.GaussianBlur(normalized_image, (3, 3), 0)
        processed_images.append(blurred_image)
        labels.append(image.label)

    return processed_images, labels
#1- features extraction
def Hog(Images):
    Features=list()
    for image in Images:
        fd, hog_image = hog(Images, orientations=9, pixels_per_cell=(3, 3),cells_per_block=(2, 2),channel_axis=0,visualize=True)
        Features.append(fd)
    return Features

    
def read_files_in_folder(folder_path):
    try:
        X_test = []
        y_test = []
        testset= []
        for label, class_name in enumerate(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    image = cv2.imread(file_path,0)
                    X_test.append(image)
                    y_test.append(class_name)
                    testset.append(Image(image,label))
        return testset,X_test,y_test    
    except OSError as e:
        print(f"Error reading files in {folder_path}: {e}")
    


trainset,validset,X_train,y_train,X_valid,y_valid=Load_Dataset()
X_train_processed, y_train_processed = preprocessing(trainset)
X_valid_processed, y_valid_processed = preprocessing(validset)
sift = cv2.SIFT_create()
X_train_hist=Hog(X_train_processed)
print("###############################")
print(len(X_train_hist[0]))
X_valid_hist=Hog(X_valid_processed)
scaler=StandardScaler()
X_train_hist=scaler.fit_transform(X_train_hist)
X_valid_hist=scaler.fit_transform(X_valid_hist)

#DimRed=PCA(n_components=64)
#X_train=DimRed.fit_transform(X_train_hist)
#X_valid=DimRed.transform(X_valid_hist)
X_train=X_train_hist
X_valid=X_valid_hist

svm = SVC(C=120,random_state=2002 ,kernel="rbf")
svm.fit(X_train, y_train_processed)

predictions = svm.predict(X_train)
print("Accuracy on Train set:", accuracy_score(y_train_processed, predictions) * 100)

# Evaluate the model
predictions = svm.predict(X_valid)
print("Accuracy on Test set:", accuracy_score(y_valid_processed, predictions) * 100)



folder_path = r"E:\4th year projects\cv\Test Samples Classification\Test Samples Classification"

testset,X_test,y_test=read_files_in_folder(folder_path)

X_test_proc,y_test_proc= preprocessing(testset)

X_test_hist=Hog(X_test_proc)

X_test_hist=scaler.transform(X_test_hist)

#X_test=DimRed.transform(X_test_hist)
X_test=X_test_hist

predictions = svm.predict(X_test)

print("Accuracy on Hidden Test set:", accuracy_score(y_test, predictions) * 100)

