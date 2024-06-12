# data_loader.py
import os
import cv2
import numpy as np
from sklearn.utils import shuffle

class DataLoader:
    def __init__(self, data_dir, augmented_data_path, image_size=(224, 224)):
        self.data_dir = data_dir
        self.augmented_data_path = augmented_data_path
        self.image_size = image_size

    def create_augmented_directories(self):
        os.makedirs(self.augmented_data_path, exist_ok=True)
        os.makedirs(f"{self.augmented_data_path}/yes", exist_ok=True)
        os.makedirs(f"{self.augmented_data_path}/no", exist_ok=True)

    def augment_images(self, file_dir, no_samples_gen, save_img_dir):
        data_gen = ImageDataGenerator(rotation_range=15, shear_range=0.1, brightness_range=(0.3, 1),
                                      horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
        for file in listdir(file_dir):
            image = cv2.imread(file_dir + '/' + file)
            image = image.reshape((1,) + image.shape)
            save_prefix = 'augmented_' + file[:-4]
            i = 0
            for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_img_dir, save_prefix=save_prefix, save_format='jpg'):
                i += 1
                if i > no_samples_gen:
                    break

    def load_data(self, dir_list):
        X = []
        y = []
        for directory in dir_list:
            for filename in listdir(directory):
                image = cv2.imread(directory + '/' + filename)
                X.append(image)
                if directory[-3:] == 'yes':
                    y.append([1])
                else:
                    y.append([0])
        X = np.array(X)
        y = np.array(y)
        X, y = shuffle(X, y)
        return X, y

    def split_data(self, X, y, test_size=0.3):
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
        return X_train, y_train, X_val, y_val, X_test, y_test
