import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils

class Utils:
    @staticmethod
    def plot_data(X, y, labels_dict, n=50):
        for index in range(len(labels_dict)):
            imgs = X[np.argwhere(y == index)][:n]
            col = 10
            i = int(n / col)
            plt.figure(figsize=(15, 6))
            c = 1
            for img in imgs:
                plt.subplot(i, col, c)
                plt.imshow(img[0])
                plt.xticks([])
                plt.yticks([])
                c += 1
            plt.suptitle('Tumor Classification {}'.format(labels_dict[index]))
            plt.show()

    @staticmethod
    def crop_brain_cnt(image, plot=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.imshow(new_image)
            plt.show()
        return new_image

    @staticmethod
    def resize_images(X, img_size=(240, 240)):
        resized_images = [cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC) / 255 for img in X]
        return np.array(resized_images)
