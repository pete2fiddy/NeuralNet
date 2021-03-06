import numpy

def get_cost(expected, results):
    subtract = numpy.subtract(results, expected)
    return 0.5 * numpy.dot(subtract, subtract)

def get_total_cost(expecteds, results):
    aggregate_error = 0
    for i in range(0, len(expecteds)):
        aggregate_error += get_cost(expecteds[i], results[i])
    return float(aggregate_error)/float(len(expecteds))

def get_partials(expected, results):
    return numpy.subtract(expected, results)
    

'''might have to multiply the partials by 1/n'''

def get_total_partials(expecteds, results):
    total_partials = numpy.zeros(numpy.asarray((expecteds)).shape[1])
    for i in range(0, len(expecteds)):
        total_partials = numpy.add(total_partials, get_partials(expecteds[i], results[i]))
    return total_partials
