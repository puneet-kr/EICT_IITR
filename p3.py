import numpy
import matplotlib.pyplot as plt

def gen_data():
    d_x = numpy.linspace(1.0, 10.0, 100)[:, numpy.newaxis]
    d_y = numpy.sin(d_x) + 0.1 * numpy.power(d_x, 2) + 0.5 * numpy.random.randn(100, 1)
    return d_x, d_y

def normalize_data(d_x, degree):
    d_x = numpy.power(d_x, range(degree))
    d_x /= numpy.max(d_x, axis=0)
    return d_x

def get_train_test(d_x, d_y):
    new_idx_list = numpy.random.permutation(len(d_x))
    ratio = 20
    test_x = d_x[new_idx_list[:ratio]]
    testY = d_y[new_idx_list[:ratio]]
    trainX = d_x[new_idx_list[ratio:]]
    trainY = d_y[new_idx_list[ratio:]]
    return test_x, testY, trainX, trainY

def get_gradient(w, x, y):
	# write your code to estimate the vectorized gradient and find mse
    return gradient, mse

def plot_graph(w,trainX,trainY,test_x,testY):
    # write your code to:
    # plot the training and testing points as per dataset on a graph
    # plot the regression line

def main():
    d_x, d_y = gen_data()
    degree = 6		# to create parameters for degree 5 function
    d_x = normalize_data(d_x, degree)
    test_x, testY, trainX, trainY = get_train_test(d_x, d_y)

    w = numpy.random.randn(degree)
    learning_rate = 0.5
    error_limit = 1e-5

    # stochastic gradient descent using mini-batches
    epochs = 1
    decay = 0.99		# can use this parameter to moderate the learning rate
    batch_size = 10
    iterations = 0
    while True:
        # loop till the error is not reduced below the error_limit value
        # find gradients and update the model parameters

	# print model parameters and final cost

    plot_graph(w,trainX,trainY,test_x,testY)

if __name__ == "__main__":
    main()
