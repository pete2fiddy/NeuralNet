from NNet.Main.Layers import Layers
import NNet.Function.Sigmoid as Sigmoid
import NNet.Function.Cost as Cost
import NNet.Function.CrossEntropy as CrossEntropy
import random
import numpy
import matplotlib.pyplot as pyplot


def get_random_function(input_range, dim_input, dim_output):
    inputs = tuple([input_range[0] + random.random()*(input_range[1]-input_range[0]) for i in range(0, dim_input)])
    outputs = tuple([random.random() for i in range(0, dim_output)])
    return inputs, outputs

def get_binary_add_functions(max_num):
    inputs = []
    outputs = []
    max_num_len = len('{0:b}'.format(max_num))
    for i in range(0, max_num-1):
        binary_num = "{0:03b}".format(i)
        binary_sum = "{0:03b}".format(i+1)
        print("num: " + str(binary_num) + " sum: " + str(binary_sum))
        vec_inputs = numpy.asarray(convert_string_to_vector(binary_num))
        vec_outputs = numpy.asarray(convert_string_to_vector(binary_sum))
           
        inputs.append(vec_inputs)
        outputs.append(vec_outputs)
    return numpy.asarray(inputs), numpy.asarray(outputs)


def convert_string_to_vector(in_string):
    vector = []
    for i in range(0, len(in_string)):
        vector.append(int(in_string[i]))
    return vector

def get_random_vector(size):
    vector = [(random.random()-.5)*2 for i in range(0, size)]
    return numpy.asarray(vector)

func_inputs, func_outputs = get_binary_add_functions(8)

print("inputs: " + str(func_inputs))
print("outputs: " + str(func_outputs))

nnet = Layers(Sigmoid, [3, 3, 3], Cost)
max_training_cycles = 1000000
learn_constant = .1
cost_threshold = .05
dropout_rate = 0
window_range = 250
costs = []
iterations = []
print(nnet)

print("initial result: " + str(nnet.get_results(func_inputs)))
print("initial cost: " + str(nnet.get_total_cost(func_outputs, nnet.get_results(func_inputs))))

stop_training = False

num_iter = 0
pyplot.ion()

pyplot.plot(costs, iterations)

while stop_training == False and num_iter < max_training_cycles:
    
    for j in range(0, len(func_inputs)):
        nnet.set_dropout_nodes(dropout_rate)
        results = nnet.get_result(func_inputs[j])
        cost = nnet.get_cost(func_outputs[j], results)
        nnet.set_node_partials(func_outputs[j], results)
        nnet.move_across_gradient(learn_constant)
        nnet.reset()
        
    if num_iter %(float(max_training_cycles)/10000.0) == 0:
        
        tot_results1 = nnet.get_results(func_inputs)
        tot_cost1 = nnet.get_total_cost(func_outputs, tot_results1)
    
        costs.append(tot_cost1)
        iterations.append(num_iter)   
        #pyplot.axis([iterations[len(iterations)-1] - window_range, iterations[len(iterations)-1],0, costs[0]]) 
        pyplot.plot(iterations, costs)
        pyplot.show()
        pyplot.pause(0.000000000000001)
        nnet.reset()
        
        if tot_cost1 < cost_threshold:
            stop_training = True
        
        print("current cost: " + str(tot_cost1))
        
    num_iter += 1



for i in range(0, 7):
    print("result of " + str(i) + ": " + str(convert_string_to_vector("{0:03b}".format(i))) + str(numpy.round(numpy.asarray(nnet.get_result(convert_string_to_vector("{0:03b}".format(i)))))))
    
while True: 
    pyplot.show()
    pyplot.pause(0.00000000001)
    
    
'''
print("final result: " + str(result2))
print("final cost: " + str(nnet.get_total_cost(func_outputs, result2)))
print(nnet)
'''
    
    
    