import numpy
import matplotlib.pyplot as plt

def esti_coeff(x, y):
    # write your code to estimate b_0, b_1 for a given x,y using formula

    return(b_0, b_1)

def plot_line(x, y, b):
    # write your code to:
    # plot the points as per dataset on a graph
    # plot the regression line

def main():
    # dataset
    x = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    y = numpy.array([60, 65, 80, 100, 110, 115, 120, 120, 130, 160])
    b = esti_coeff(x, y)
    plot_line(x, y, b)

if __name__ == "__main__":
    main()
