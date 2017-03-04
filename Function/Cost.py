import numpy

def get_cost(expected, results):
    sum = 0
    for i in range(0, len(expected)):
        sum += (expected[i]-results[i])**2
    return sum * 0.5
    #return numpy.linalg.norm( numpy.subtract(numpy.asarray(expected), numpy.asarray(results))**2 )

def get_partials(expected, results):
    partials = numpy.zeros(numpy.asarray((expected)).shape[0])
    for i in range(0, partials.shape[0]):
        partials[i] = results[i] - expected[i]
    return partials