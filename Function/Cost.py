import numpy

def get_cost(expected, results):
    sum = 0
    for i in range(0, len(expected)):
        sum += (expected[i]-results[i])**2
    return sum * 0.5
    #return numpy.linalg.norm( numpy.subtract(numpy.asarray(expected), numpy.asarray(results))**2 )

def get_total_cost(expecteds, results):
    aggregate_error = 0
    for i in range(0, len(expecteds)):
        aggregate_error += get_cost(expecteds[i], results[i])
    return aggregate_error

def get_partials(expected, results):
    partials = numpy.zeros(numpy.asarray((expected)).shape[0])
    for i in range(0, partials.shape[0]):
        partials[i] = results[i] - expected[i]
    return partials

def get_total_partials(expecteds, results):
    total_partials = numpy.zeros(numpy.asarray((expecteds)).shape[1])
    for i in range(0, len(expecteds)):
        total_partials = numpy.add(total_partials, get_partials(expecteds[i], results[i]))
    return total_partials
        