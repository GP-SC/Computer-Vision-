import numpy as np
import cv2
import os
import glob
from PIL import Image

def preprocessing(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_width, new_height = 300, 200
    resized_image = cv2.resize(gray_image, (new_width, new_height))
    normalized_image = resized_image / 255.0
    cv2.imshow('Original Image', img)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.imshow('Resized Image', resized_image)
    cv2.imshow('Normalized Image', normalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def readtrainimage(dataset_path):
 # You may need to install the Pillow library for image processing

    # Specify the path to the root folder
    root_path = dataset_path

    # Specify the main folders
    data_folder = "Train"
    classification_product_folder = "Product Classification"

    # Get the list of product folders
    product_folders = [folder for folder in os.listdir(os.path.join(root_path, classification_product_folder)) if
                       os.path.isdir(os.path.join(root_path, classification_product_folder, folder))]

    # Iterate through each product folder
    for product_folder in product_folders:
        product_path = os.path.join(root_path, classification_product_folder, product_folder)

        # Check for the existence of the training folder
        training_folder = os.path.join(product_path, data_folder, "training")

        if os.path.exists(training_folder):
            # Get a list of image files in the training folder
            image_files = [os.path.join(training_folder, file) for file in os.listdir(training_folder) if
                           file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            # Iterate through each image file in the training folder
            for image_file in image_files:
                # You can now perform operations on each image
                # For example, open the image using Pillow
                img = Image.open(image_file)

                # Add your image processing or analysis code here

                # Optionally, you can display the image
                img.show()
        else:
            print(f"Warning: Training folder missing for product '{product_folder}'.")

    # Note: Adjust the file extensions in the endsWith tuple based on the actual file formats of your images.

image_path="Data\Product Classification"
original_image = cv2.imread(image_path)
dataset_path = "Data"
readtrainimage(dataset_path)
preprocessing(original_image)





# Get the list of product folders
