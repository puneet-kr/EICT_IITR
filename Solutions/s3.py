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
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0/len(x))*numpy.sum(numpy.power(error, 2))
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, mse

def plot_graph(w,trainX,trainY,test_x,testY):
    y_model = numpy.polyval(w[::-1], numpy.linspace(0,1,100))
    plt.plot(numpy.linspace(0,1,100), y_model, c='g', label='Model')
    plt.scatter(trainX[:,1], trainY, c='b', label='Train Set')
    plt.scatter(test_x[:,1], testY, c='r', label='Test Set')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0,1)
    plt.show()

def main():
    d_x, d_y = gen_data()
    degree = 6
    d_x = normalize_data(d_x, degree)
    test_x, testY, trainX, trainY = get_train_test(d_x, d_y)

    w = numpy.random.randn(degree)
    learning_rate = 0.5
    error_limit = 1e-5

    # stochastic gradient descent using mini-batches
    epochs = 1
    decay = 0.99
    batch_size = 10
    iterations = 0
    while True:
        new_idx_list = numpy.random.permutation(len(trainX))
        trainX = trainX[new_idx_list]
        trainY = trainY[new_idx_list]
        b=0
        while b < len(trainX):
            tx = trainX[b : b+batch_size]
            ty = trainY[b : b+batch_size]
            gradient = get_gradient(w, tx, ty)[0]
            error = get_gradient(w, trainX, trainY)[1]
            w -= learning_rate * gradient
            iterations += 1
            b += batch_size

        if epochs%100==0:
            new_error = get_gradient(w, trainX, trainY)[1]
            print ("Epoch: %d - Error: %.4f" %(epochs, new_error))

            if abs(new_error - error) < error_limit:
                print ("Converged.")
                break

        learning_rate = learning_rate * (decay ** int(epochs/1000))
        epochs += 1

    print ("w =",w)
    print ("Squared error =", get_gradient(w, test_x, testY)[1])
    print ("Total iterations =", iterations)

    plot_graph(w,trainX,trainY,test_x,testY)

if __name__ == "__main__":
    main()
