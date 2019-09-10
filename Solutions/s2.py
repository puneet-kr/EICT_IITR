import numpy
import matplotlib.pyplot as plt

def gen_data():
    d_x = numpy.linspace(0, 5, 100)[:, numpy.newaxis]
    d_y = numpy.sin(d_x) + 10 * numpy.sin(d_x) + 0.5 * numpy.random.randn(100, 1)
    d_x /= numpy.max(d_x)
    return d_x, d_y

def get_train_test(d_x, d_y):
    new_idx_list = numpy.random.permutation(len(d_x))
    ratio = 20
    testX = d_x[new_idx_list[:ratio]]
    testY = d_y[new_idx_list[:ratio]]
    trainX = d_x[new_idx_list[ratio:]]
    trainY = d_y[new_idx_list[ratio:]]
    return testX, testY, trainX, trainY

def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0/len(x))*numpy.sum(numpy.power(error, 2))
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, mse

def plot_graph(d_x, w, trainX, trainY, testX, testY):
    plt.plot(d_x[:,1], d_x.dot(w), c='g', label='Model')
    plt.scatter(trainX[:,1], trainY, c='b', label='Train Set')
    plt.scatter(testX[:,1], testY, c='r', label='Test Set')
    plt.grid()
    plt.legend(loc=2)
    plt.xlim(0,1.05)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.show()

def main():
    # dataset
    d_x, d_y = gen_data()
    d_x = numpy.hstack((numpy.ones_like(d_x), d_x))
    testX, testY, trainX, trainY = get_train_test(d_x, d_y)

    w = numpy.random.randn(2)
    learning_rate = 0.5
    err_limit = 1e-5

    # gradient descent
    iterations = 1
    while True:
        gradient, error = get_gradient(w, trainX, trainY)
        new_w = w - learning_rate * gradient

        # termination criteria
        if numpy.sum(abs(new_w - w)) < err_limit:
            print ("Converged.")
            break

        if iterations % 100 == 0:
            print ("Iteration: %d - Error: %.4f" %(iterations, error))

        iterations += 1
        w = new_w

    print ("w =", w)
    print ("Squared error =", get_gradient(w, testX, testY)[1])

    plot_graph(d_x, w, trainX, trainY, testX, testY)

if __name__ == "__main__":
    main()
