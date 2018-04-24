from keras.layers import Input,Dense,Convolution2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as  np
import matplotlib.pyplot as plt

input_img = Input(shape = (28,28,1))
x = Convolution2D(16,3,3,activation = 'relu',border_mode = 'same')(input_img)
x = MaxPooling2D((2,2),border_mode = 'same')(x)
x = Convolution2D(8,3,3,activation = 'relu',border_mode = 'same')(x)
x = MaxPooling2D((2,2),border_mode = 'same')(x)
x = Convolution2D(8,3,3,activation = 'relu',border_mode = 'same')(x)
encoded = MaxPooling2D((2,2),border_mode = 'same')(x)

#at this point the representation is (8,4,4) i.e. 128-dimensional

x = Convolution2D(8,3,3,activation = 'relu',border_mode = 'same')(encoded)
x = UpSampling2D((2,2))(x)
x = Convolution2D(8,3,3,activation = 'relu',border_mode = 'same')(x)
x = UpSampling2D((2,2))(x)
x = Convolution2D(16,3,3,activation= 'relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Convolution2D(1,3,3,activation = 'sigmoid',border_mode = 'same')(x)

autoencoder = Model(input_img,decoded)
autoencoder.compile(optimizer = Adam(),loss = 'binary_crossentropy')

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train,(len(x_train),28,28,1))
x_test = np.reshape(x_test,(len(x_test),28,28,1))

autoencoder.fit(x_train,x_train,nb_epoch = 20,batch_size = 128,shuffle = True,validation_data = (x_test,x_test))

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

autoencoder.summary()

model_extractfeatures = Model(input=autoencoder.input, output=autoencoder.get_layer('maxpooling2d_3').output)
encoded_imgs = model_extractfeatures.predict(x_test)
print(encoded_imgs.shape)
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

