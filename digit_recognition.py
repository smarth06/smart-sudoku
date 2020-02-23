import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

class Recognizer():
    def __init__(self,input_shape, filters, kernel_size, pool_size, number_neurons_hidden, rate, number_neurons_output, epochs):
        self.input_shape = input_shape #(28, 28, 1)
        self.filters = filters # 28
        self.kernel_size = kernel_size # (3, 3)
        self.pool_size = pool_size # (2, 2)
        self.number_neurons_hidden = number_neurons_hidden # (128)
        self.rate = rate # 0.2
        self.number_neurons_output = number_neurons_output #10
        self.epochs = epochs

    def initialize_cnn(self):
        model = Sequential()
        ## Conv2D
        ## filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        model.add(Conv2D(self.filters, kernel_size = self.kernel_size, input_shape = self.input_shape))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Conv2D(16, (3,3), input_shape = self.input_shape))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(self.number_neurons_hidden, activation='relu'))
        model.add(Dropout(self.rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.number_neurons_output,activation=tf.nn.softmax))
        return model

    def fit_model(self, model, x_train, y_train):
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
        model.fit(x = x_train, y = y_train, epochs = self.epochs)
        return model

    def evaluate_model(self, model, x_test, y_test):
        return model.evaluate(x_test, y_test)
        
    def predict(self, model, x_test, y_test, image_index, img_rows, img_cols):
        plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
        pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
        print(pred.argmax())