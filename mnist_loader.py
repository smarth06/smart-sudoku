import tensorflow as tf
import matplotlib.pyplot as plt

def load_mnist_data():
    return tf.keras.datasets.mnist.load_data()

def visualize_data(x_train, y_train):
    image_index = 0
    print(y_train[image_index])
    plt.imshow(x_train[image_index],cmap='Greys')
    plt.show()

def normalize_data(x_train, x_test):
    '''
        As Keras API expects the data to be 4 dimensional and our data is of
        3 dimensions.
        This function also normalizes our data by dividing RGB codes to 255.
    '''
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train/255
    x_test = x_test/255

    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    return x_train,x_test