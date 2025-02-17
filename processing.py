import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

class ImageDataLoader:
    def __init__(self, image_path, image_size=150):
        self.image_path = image_path
        self.image_size = image_size
        self.image_data = []
        self.label_data = []
        self.label_name = os.listdir(image_path)

    def preprocess_data(self):
        print("----------------Preprocessing train data-----------------\n")
        
        for label in self.label_name:
            data_path = os.path.join(self.image_path, label)
            for filename in tqdm(os.listdir(data_path)):
                image = cv2.imread(os.path.join(data_path, filename))
                image = cv2.resize(image, (self.image_size, self.image_size))
                
                self.image_data.append(image)
                self.label_data.append(label)

        self.image_data = np.array(self.image_data)
        self.label_data = np.array(self.label_data)

    def shuffle_and_split(self, test_size=0.3, random_state=42):
        self.image_data, self.label_data = shuffle(self.image_data, self.label_data, random_state=random_state)
        X_train, X_test, Y_train, Y_test = train_test_split(self.image_data, self.label_data, test_size=test_size, random_state=random_state)
        
        Y_train = self._convert_labels_to_categorical(Y_train)
        Y_test = self._convert_labels_to_categorical(Y_test)
        
        return X_train, X_test, Y_train, Y_test

    def _convert_labels_to_categorical(self, labels):
        new_labels = [self.label_name.index(label) for label in labels]
        return to_categorical(new_labels)
