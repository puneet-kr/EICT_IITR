import numpy
import matplotlib.pyplot as plt

def esti_coeff(x, y):
    # size of the dataset
    n = numpy.size(x)

    u_x, u_y = numpy.mean(x), numpy.mean(y)

    # cross-deviation and deviation about x
    SS_xy = numpy.sum(y*x - n*u_y*u_x)
    SS_xx = numpy.sum(x*x - n*u_x*u_x)

    # regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = u_y - b_1*u_x

    return b_0, b_1

def plot_line(x, y, b):
    plt.scatter(x, y, color = "m",marker = "o", s = 30)     # plot points on a graph
    y_pred = b[0] + b[1]*x      # predicted vector
    plt.plot(x, y_pred, color = "g")    # plot line
    plt.xlabel('Size')
    plt.ylabel('Cost')
    plt.show()

def main():
    # dataset
    x = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    y = numpy.array([60, 65, 80, 100, 110, 115, 120, 120, 130, 160])

    # estimate coefficients
    b = esti_coeff(x, y)
    print("Coefficients are: b_0 = {} b_1 = {}".format(b[0], b[1]))

    plot_line(x, y, b)

if __name__ == "__main__":
    main()
