import solveSudoku
import digit_recognition
import mnist_loader

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist_loader.load_mnist_data()
    print(x_train)
    print(y_train)
    print(len(x_train))
    print(len(x_test))

    mnist_loader.visualize_data(x_train, y_train, x_test, y_test)

    print(x_train.shape) ## shape of training data size = 60000 * 28 * 28

    