import keras

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

(x_train, y_train), (x_test, y_test) = mnist.load_data()#get data

num_classes = 10
num_training=58000
num_test=1000

mask = list(range(num_training,x_train.shape[0]))#last 2000 for a finel test
x_test2 = x_train[mask]
y_test2 = y_train[mask]

mask = list(range(num_training)) #58000 images for training
x_train = x_train[mask]
y_train = y_train[mask]


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)#reshape for CNN
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train/255#scaling
x_test = x_test/255
print(y_train.shape)#finel shape
print(x_train.shape)


model = Sequential ([
    Conv2D(9,kernel_size=(3,3),activation='relu',input_shape=(28, 28, 1),padding='same'),
    MaxPooling2D(pool_size=(2,2),strides=1),
    Conv2D(14,kernel_size=(5,5),activation='relu',input_shape=(28, 28, 1),padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=1),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')

    # Flatten(input_shape=(28, 28)),
    # Dense(units=512, activation='relu'),
    # Dropout(0.5),
    # Dense(10, activation='softmax')
])
model.compile(loss=keras.losses.categorical_crossentropy,optimizer="adam",metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=100,epochs=7,verbose=1,validation_data=(x_test, y_test))
print(model.summary())


print("The model has successfully trained")
model.save('mywork2.h5')
print("Saving the model as mywork2.h5")


