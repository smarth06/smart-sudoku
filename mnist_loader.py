import tensorflow as tf
import matplotlib.pyplot as plt

def load_mnist_data():
    return tf.keras.datasets.mnist.load_data()

def visualize_data(x_train, y_train, x_test, y_test):
    image_index = 0
    print(y_train[image_index])
    plt.imshow(x_train[image_index],cmap='Greys')
    plt.show()