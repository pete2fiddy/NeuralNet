import numpy
from math import log1p

def get_cost(expected, results):
    sum = 0
    for i in range(0, len(expected)):
        sum += expected[i]*log1p(results[i]) + ((1-expected[i]) * log1p(1-results[i]))
    return sum

def get_total_cost(expecteds, results):
    aggregate_error = 0
    for i in range(0, len(expecteds)):
        aggregate_error += get_cost(expecteds[i], results[i])
    return (1.0/float(len(expecteds)))*aggregate_error

def get_partials(expected, results):
    return numpy.multiply(results, numpy.subtract(results, expected))
    #return numpy.subtract(expected, results)
    
    
def get_total_partials(expecteds, results):
    total_partials = numpy.zeros(numpy.asarray((expecteds)).shape[1])
    for i in range(0, len(expecteds)):
        total_partials = numpy.add(total_partials, get_partials(expecteds[i], results[i]))
    return total_partials * (1.0/float(len(expecteds)))

