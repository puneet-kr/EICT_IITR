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
	# write your code to estimate the vectorized gradient and find mse
    return gradient, mse

def plot_graph(d_x, w, trainX, trainY, testX, testY):
    # write your code to:
    # plot the training and testing points as per dataset on a graph
    # plot the regression line

def main():
    # datasets
    d_x, d_y = gen_data()
    d_x = numpy.hstack((numpy.ones_like(d_x), d_x))
    testX, testY, trainX, trainY = get_train_test(d_x, d_y)

    w = numpy.random.randn(2)
    learning_rate = 0.5		# learning rate
    err_limit = 1e-5	# termination limit of error

    # gradient descent
    iterations = 1
    while True:
        # loop till the error is not reduced below the err_limit value
        # find gradients and update the model parameters

	# print model parameters and final mse error

    plot_graph(d_x, w, trainX, trainY, testX, testY)

if __name__ == "__main__":
    main()
