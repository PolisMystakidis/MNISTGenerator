import tensorflow as tf
from matplotlib import pyplot as plt
from Utilities import flatten_images , unflatten_images
import time

from Autoencoder import autoencoder
from random import randint , random

mnist = tf.keras.datasets.mnist
(train_img_original, train_label), (test_img_original, test_label) = mnist.load_data()
train_img_original = train_img_original / 255
test_img_original = test_img_original / 255

# Flatten
train_img, test_img = flatten_images(train_img_original, test_img_original)

# -----Making denoiser with autoencoder------

corrupted_train = train_img.copy()
corrupted_test = test_img.copy()

# Corrupt train
for i in range(len(corrupted_train)):
    for _ in range(100):
        pos = randint(0,784-1)
        val = random()
        corrupted_train[i][pos] = val
# Corrupt test
for i in range(len(corrupted_test)):
    for _ in range(100):
        pos = randint(0,784-1)
        val = random()
        corrupted_test[i][pos] = val

corrupted_train_imgs , corrupted_test_imgs = unflatten_images(corrupted_train,corrupted_test)

plt.imshow(corrupted_train_imgs[0])
plt.show()

denoiser = autoencoder(list_enc_shapes=[584,256,144],list_dec_shapes=[256,584,784])
denoiser .generate_autoencoder()
t0  = time.time()
denoiser.fit_autoencoder(corrupted_train,train_img,corrupted_test,test_img,epochs=100,batchsize=32)
print("Training time : ",time.time()-t0)

denoised_train = denoiser.model.predict(corrupted_train)
denoised_test = denoiser.model.predict(corrupted_test)

denoised_train_imgs , denoised_test_imgs = unflatten_images(denoised_train,denoised_test)


img_index = 1
plt.subplot(1,3,1)
plt.imshow(train_img_original[img_index])
plt.subplot(1,3,2)
plt.imshow(corrupted_train_imgs[img_index])
plt.subplot(1,3,3)
plt.imshow(denoised_train_imgs[img_index])
plt.show()