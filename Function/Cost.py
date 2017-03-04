import numpy

def get_cost(expected, results):
    for result_index in range(0, len(results)):
        sum = 0
        for i in range(0, len(expected[0])):
            sum += (expected[result_index][i]-results[result_index][i])**2
    return sum * 0.5
    #return numpy.linalg.norm( numpy.subtract(numpy.asarray(expected), numpy.asarray(results))**2 )

def get_partials(expected, results):
    partials = numpy.zeros(numpy.asarray((expected)).shape[1])
    for i in range(0, partials.shape[0]):
        for j in range(0, len(expected)):
            partials[i] += results[j][i] - expected[j][i]
    return partials