import tensorflow as tf
import numpy as np
from random import randint , random
from matplotlib import pyplot as plt
import time
from Utilities import flatten_images , unflatten_images

from Autoencoder import autoencoder

mnist = tf.keras.datasets.mnist
(train_img_original, train_label), (test_img_original, test_label) = mnist.load_data()
train_img_original = train_img_original / 255
test_img_original = test_img_original / 255

# Flatten
train_img, test_img = flatten_images(train_img_original, test_img_original)
print(train_img.shape)

train_label_onehot = tf.one_hot(train_label,10).numpy()
test_label_onehot = tf.one_hot(test_label,10).numpy()


# Use autoencoder to find the decoder which goes from 10 to 784 dims
data_generator = autoencoder(list_enc_shapes=[584,256,10],list_dec_shapes=[584,784])
data_generator .generate_autoencoder()
#t0 = time.time()
#data_generator.fit_autoencoder(train_img,train_img,test_img,test_img,epochs=100,batchsize=32)
#print("Autoencoder training : ",time.time()-t0)
#data_generator.save("data_generator")

data_generator.load("data_generator")

predictions_train = data_generator.encode(train_img)
predictions_test = data_generator.encode(test_img)

# Use a newral network which learns how to transform onehot encoded inputs of image labels to the output of the encoded image
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation="sigmoid")
])

model.compile(optimizer='adam',
              loss='mse')
#t0=time.time()
#model.fit(train_label_onehot, predictions_train, epochs=100, batch_size=32, shuffle=True, validation_data=(test_label_onehot,predictions_test))
#print("Model training : ", time.time()-t0)

#model.save_weights("translator.h5f")

model.load_weights("translator.h5f")


# Pass onehot encoded data from model and then from the decoder to generate image
pred = data_generator.decode(model.predict(test_label_onehot))
pred_img = np.resize(pred,(len(pred),28,28))
i=0
for im in pred_img[:10]:
    plt.title("Label : "+str(test_label[i]))
    plt.imshow(im)
    plt.show()
    i+=1


corrupted_labels = train_label_onehot.copy()
noise = np.random.rand(corrupted_labels.shape[0],corrupted_labels.shape[1])/2
corrupted_labels = corrupted_labels+noise

pred_new = data_generator.decode(model.predict(corrupted_labels))
pred_new_img = np.resize(pred_new , ( len(pred_new),28,28))
i=0
for im in pred_new_img[:10]:
    plt.title("Label : "+str(train_label[i]))
    plt.imshow(im)
    plt.show()
    i+=1