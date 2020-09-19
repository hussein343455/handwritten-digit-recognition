import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import keras
from keras.datasets import mnist
from keras.engine.saving import load_model

model = load_model('mywork2.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10
num_training=58000

mask = list(range(num_training,x_train.shape[0]))
x_test2 = x_train[mask]
y_test2 = y_train[mask]
print(x_test2.shape)
print(y_test2.shape)

x_test2 = x_test2.reshape(x_test2.shape[0], 28, 28, 1)
y_test2 = keras.utils.to_categorical(y_test2, num_classes)

score = model.evaluate(x_test2, y_test2, verbose = 1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])