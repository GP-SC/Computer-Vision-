import numpy as np
import cv2
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
class Image:
    def __init__(self,img,label:str):
        self.img = img
        self.label=label
        self.pred=None
    def SetPred(self,pred):
        self.pred=pred
def preprocessing(DataSet:list(Image)):
    for image in DataSet:
        img=image.img

        '''new_width, new_height = 300, 200
        resized_image = cv2.resize(gray_image, (new_width, new_height))
        normalized_image = resized_image / 255.0
'''
def Load_Dataset():
    currpath="Data/Product Classification"
    Classes=os.listdir(currpath)
    Classes.sort( key=lambda x: int(x))
    Classes=list(map(lambda x:currpath+"/"+x,Classes))
    X_train,y_train,X_valid,y_valid=list()
    trainset=list()
    validset=list()
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


trainset,validset,X_train,y_train,X_valid,y_valid=Load_Dataset()






# Get the list of product folders
