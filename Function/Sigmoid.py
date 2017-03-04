import numpy


def func(x):
    return 1.0/(1 + numpy.exp(-x))

#returns the derivative of the function in terms of the function's actual output at that point
def dfunc(x):
    return x*(1.0-x)