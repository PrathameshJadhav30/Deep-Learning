
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = "D:\Deep Learning/05-Experiment-5/traffic/traffic_Data/DATA"  
classes = os.listdir(data_dir)

data = []
labels = []

for i, folder in enumerate(classes):
    folder_path = os.path.join(data_dir, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (30, 30))
            data.append(image)
            labels.append(int(folder))
        except:
            print(f"Error loading image: {img_path}")

X = np.array(data)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)

print("Data preprocessing completed. Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
