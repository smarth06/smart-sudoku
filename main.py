import solveSudoku
from digit_recognition import Recognizer
import mnist_loader

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist_loader.load_mnist_data()
    mnist_loader.visualize_data(x_train, y_train)
    print(x_train.shape) ## shape of training data size = 60000 * 28 * 28
    x_train ,x_test = mnist_loader.normalize_data(x_train, x_test)

    input_shape = (28, 28, 1)
    filters = 28
    kernel_size = (3, 3)
    pool_size = (2, 2)
    number_neurons_hidden = 128
    rate = 0.2
    number_neurons_output = 10
    epochs = 10
    
    recognizer = Recognizer(input_shape, filters, kernel_size, pool_size, number_neurons_hidden, rate, number_neurons_output, epochs)
    model = recognizer.initialize_cnn()
    model = recognizer.fit_model(model, x_train, y_train)
    print(recognizer.evaluate_model(model, x_test, y_test))

    image_index = 3333
    img_rows = 28
    img_cols = 28
    recognizer.predict(model, x_test, y_test, image_index, img_rows, img_cols)