import tensorflow as tf
import numpy as np
from random import randint , random
from matplotlib import pyplot as plt
import time
from Utilities import flatten_images , unflatten_images


class autoencoder():
    def __init__(self,list_enc_shapes =[], list_dec_shapes=[] ):
        self.encoders = [tf.keras.layers.Dense(x, activation='sigmoid') for x in list_enc_shapes]
        self.decoders = [tf.keras.layers.Dense(x, activation='sigmoid') for x in list_dec_shapes]

        self.model = tf.keras.Sequential(self.encoders + self.decoders)
        self.encoder = tf.keras.Sequential(self.encoders)
        self.decoder = tf.keras.Sequential(self.decoders)
    def generate_autoencoder(self):


        self.model.compile(optimizer='adam',
                      loss="mse")
        return self.encoder , self.decoder
    def fit_autoencoder(self,data,labels,val_data,val_labels,epochs,batchsize):
        self.model.fit(data, labels, epochs=epochs, batch_size=batchsize, shuffle=True, validation_data=(val_data, val_labels))

    def encode(self,data):
        return self.encoder.predict(data)

    def decode(self,data):
        return self.decoder.predict(data)






mnist = tf.keras.datasets.mnist
(train_img_original, train_label), (test_img_original, test_label) = mnist.load_data()
train_img_original = train_img_original / 255
test_img_original = test_img_original / 255



# Flatten
train_img , test_img = flatten_images(train_img_original,test_img_original)
print(train_img.shape)

#-------Making simple autoencoder--------
# ae = autoencoder(list_enc_shapes=[584,256,144],list_dec_shapes=[256,584,784])
# ae .generate_autoencoder()
#
#
#
# t0  = time.time()
# ae.fit_autoencoder(train_img,train_img,test_img,test_img,epochs=5,batchsize=32)
# print("Training time : ",time.time()-t0)
#
#
#
# x = ae.encode(np.asarray([train_img[0]]))[0]
# im_encoded = np.resize(x ,(12,12))
# im = ae.decode(np.asarray([x]))[0]
# im = np.resize(im ,(28,28))
#
# plt.subplot(1,3,1)
# plt.imshow(train_img_original[0])
# plt.subplot(1,3,2)
# plt.imshow(im_encoded)
# plt.subplot(1,3,3)
# plt.imshow(im)
# plt.show()

# -----Making denoiser with autoencoder------

# corrupted_train = train_img.copy()
# corrupted_test = test_img.copy()
#
# # Corrupt train
# for i in range(len(corrupted_train)):
#     for _ in range(100):
#         pos = randint(0,784-1)
#         val = random()
#         corrupted_train[i][pos] = val
# # Corrupt test
# for i in range(len(corrupted_test)):
#     for _ in range(100):
#         pos = randint(0,784-1)
#         val = random()
#         corrupted_test[i][pos] = val
#
# corrupted_train_imgs , corrupted_test_imgs = unflatten_images(corrupted_train,corrupted_test)
#
# plt.imshow(corrupted_train_imgs[0])
# plt.show()
#
# denoiser = autoencoder(list_enc_shapes=[584,256,144],list_dec_shapes=[256,584,784])
# denoiser .generate_autoencoder()
# t0  = time.time()
# denoiser.fit_autoencoder(corrupted_train,train_img,corrupted_test,test_img,epochs=10,batchsize=32)
# print("Training time : ",time.time()-t0)
#
# denoised_train = denoiser.model.predict(corrupted_train)
# denoised_test = denoiser.model.predict(corrupted_test)
#
# denoised_train_imgs , denoised_test_imgs = unflatten_images(denoised_train,denoised_test)
#
#
# img_index = 1
# plt.subplot(1,3,1)
# plt.imshow(train_img_original[img_index])
# plt.subplot(1,3,2)
# plt.imshow(corrupted_train_imgs[img_index])
# plt.subplot(1,3,3)
# plt.imshow(denoised_train_imgs[img_index])
# plt.show()

# ---------Making a data generator----------
train_label_onehot = tf.one_hot(train_label,10).numpy()
test_label_onehot = tf.one_hot(test_label,10).numpy()


data_generator = autoencoder(list_enc_shapes=[584,256,10],list_dec_shapes=[256,584,784])
data_generator .generate_autoencoder()

# # # Training with original data
# t0  = time.time()
# data_generator.fit_autoencoder(train_img,train_img,test_img,test_img,epochs=10,batchsize=100)
# print("Training time : ",time.time()-t0)
#
# pred = data_generator.decode(test_label_onehot)
# pred_img = np.resize(pred,(len(pred),28,28))
# i=0
# for im in pred_img[:10]:
#     plt.title("Label : "+str(test_label[i]))
#     plt.imshow(im)
#     plt.show()
#     i+=1



# # Training with corrupted data
#
# data_generator_corr = autoencoder(list_enc_shapes=[584,256,144],list_dec_shapes=[256,584,784])
# data_generator_corr .generate_autoencoder()
#
# # Training with original data
# t0  = time.time()
# data_generator_corr.fit_autoencoder(corrupted_train,train_img,corrupted_test,test_img,epochs=100,batchsize=32)
# print("Training time : ",time.time()-t0)
#
# pred = data_generator.decode(test_label_onehot)
# pred_img = np.resize(pred,(len(pred),28,28))
# i=0
# for im in pred_img[:10]:
#     plt.title("Label : "+str(test_label[i]))
#     plt.imshow(im)
#     plt.show()
#     i+=1

# # Recreate digit images from labels
####-------------

# Use autoencoder to find the decoder which goes from 10 to 784 dims



data_generator = autoencoder(list_enc_shapes=[584,256,10],list_dec_shapes=[584,784])
data_generator .generate_autoencoder()
data_generator.fit_autoencoder(train_img,train_img,test_img,test_img,epochs=25,batchsize=32)

predictions_train = data_generator.encode(train_img)
predictions_test = data_generator.encode(test_img)

# Use a newral network which learns how to transform onehot encoded inputs of image labels to the output of the encoded image
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation="sigmoid")
])

model.compile(optimizer='adam',
              loss='mse')
model.fit(train_label_onehot, predictions_train, epochs=25, batch_size=100, shuffle=True, validation_data=(test_label_onehot,predictions_test))

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
noise = np.random.rand(corrupted_labels.shape[0],corrupted_labels.shape[1])/5
corrupted_labels = corrupted_labels+noise

pred_new = data_generator.decode(model.predict(corrupted_labels))
pred_new_img = np.resize(pred_new , ( len(pred_new),28,28))
i=0
for im in pred_new_img[:10]:
    plt.title("Label : "+str(train_label[i]))
    plt.imshow(im)
    plt.show()
    i+=1





