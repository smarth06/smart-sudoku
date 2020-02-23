import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np

class SudokuDataLoader():
    def __init__(self, dir, data_type):
        self.dir = dir
        self.data_type = data_type
    
    def load_data(self):
        X_train = []
        X_test = []
        count = 0
        train = self.data_type[0]
        train_path = os.path.join(self.dir,train)
        for img in (os.listdir(train_path)):  
            if img.endswith('.jpg'):
                print(img)
                img_array = cv2.imread(os.path.join(train_path,img), cv2.IMREAD_GRAYSCALE)
                X_train.append(img_array)
                count = count + 1
            if(count>0):
                break

        count = 0
        test = self.data_type[1]
        test_path = os.path.join(self.dir,test)
        for img in (os.listdir(test_path)):  
            if img.endswith('.jpg'):
                img_array = cv2.imread(os.path.join(test_path,img), cv2.IMREAD_GRAYSCALE)
                X_test.append(img_array)
                count = count + 1
            if(count>0):
                break

        return np.asarray(X_train), np.asarray(X_test)

