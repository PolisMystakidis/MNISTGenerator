import numpy as np


# load dataset
def flatten_images(train_img,test_img):
    return np.resize(train_img, (60000, 28 * 28)) , np.resize(test_img, (10000, 28 * 28))
def unflatten_images(train_img,test_img):
    return np.resize(train_img, (60000, 28 , 28)), np.resize(test_img, (10000, 28 , 28))