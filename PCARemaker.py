from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
def mean_square_error(original,reproduced):
    diff = original - reproduced
    return sum(sum(diff*diff)/diff.shape[1]) / diff.shape[0]



# load dataset
def flatten_images(train_img,test_img):
    return np.resize(train_img, (60000, 28 * 28)) , np.resize(test_img, (10000, 28 * 28))
def unflatten_images(train_img,test_img):
    return np.resize(train_img, (60000, 28 , 28)), np.resize(test_img, (10000, 28 , 28))

mnist = tf.keras.datasets.mnist
(train_img_original, train_label), (test_img_original, test_label) = mnist.load_data()
train_img_original = train_img_original / 255
test_img_original = test_img_original / 255



# Flatten
train_img , test_img = flatten_images(train_img_original,test_img_original)
print(train_img.shape)

pca = PCA(n_components=144)
t0 = time.time()
pca.fit(train_img)
print("Time : ",time.time()-t0)

reduced = pca.transform(train_img)
reduced_imgs = np.resize(reduced, (60000, 28 , 28))

reproduced = pca.inverse_transform(reduced)
reproduced_imgs = np.resize(reproduced, (60000, 28 , 28))

print("Mean squared error :",mean_square_error(train_img,reproduced))

plt.subplot(1,3,1)
plt.imshow(train_img_original[0])
plt.subplot(1,3,2)
plt.imshow(reduced_imgs[0])
plt.subplot(1,3,3)
plt.imshow(reproduced_imgs[0])
plt.show()

