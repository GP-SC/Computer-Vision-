from PIL import Image
import random
import numpy as np
import os 
import cv2
# Open the base image where you want to place other images
def DataAugment(imgpath):
    base_image = Image.new('RGBA', (1200, 1200))

    # Open the image you want to place on the base image
    image_to_place = Image.open(imgpath).convert("RGBA")
    
    image_to_place = image_to_place.rotate(np.random.randint(0,180)) 
    # Get the width and height of the base image
    base_width, base_height = base_image.size

    # Get the width and height of the image to be placed
    place_width, place_height = image_to_place.size

    # Define the number of times you want to place the image randomly
    

    # Define the maximum x and y coordinates for placing the image
    max_x = base_width - place_width
    max_y = base_height - place_height

    x = int(random.randint(0, max_x)/9*np.random.randint(0,9))
    y = int(random.randint(0, max_y)/9*np.random.randint(0,9))
    # Paste the image onto the base image at the randomly generated coordinates
    base_image.paste(image_to_place, (x, y), image_to_place)

    # Save the resulting image
    print(imgpath[:-4]+"Augmented.png")
    base_image.save(imgpath[:-4]+"Augmented.png")

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
            if("Augmented" in file):continue
            DataAugment(file)

Load_Dataset()