import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from Utilities import flatten_images , unflatten_images
import time

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
    def save(self,name):
        self.model.save_weights(name+".h5f")
    def load(self,name):
        self.model.load_weights(name+".h5f")
        self.encoder = tf.keras.Sequential(self.encoders)
        self.decoder = tf.keras.Sequential(self.decoders)


if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist
    (train_img_original, train_label), (test_img_original, test_label) = mnist.load_data()
    train_img_original = train_img_original / 255
    test_img_original = test_img_original / 255

    # Flatten
    train_img, test_img = flatten_images(train_img_original, test_img_original)
    print(train_img.shape)


    # -------Making simple autoencoder--------
    ae = autoencoder(list_enc_shapes=[584, 256, 144], list_dec_shapes=[256, 584, 784])
    ae.generate_autoencoder()

    t0 = time.time()
    ae.fit_autoencoder(train_img, train_img, test_img, test_img, epochs=100, batchsize=32)
    print("Training time : ", time.time() - t0)

    x = ae.encode(np.asarray([train_img[0]]))[0]
    im_encoded = np.resize(x, (12, 12))
    im = ae.decode(np.asarray([x]))[0]
    im = np.resize(im, (28, 28))

    plt.subplot(1, 3, 1)
    plt.imshow(train_img_original[0])
    plt.subplot(1, 3, 2)
    plt.imshow(im_encoded)
    plt.subplot(1, 3, 3)
    plt.imshow(im)
    plt.show()