from NNet.Main.Layers import Layers
import NNet.Function.Sigmoid as Sigmoid
import NNet.Function.Cost as Cost
import random
import numpy


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
        inputs.append(convert_string_to_vector(binary_num))
        outputs.append(convert_string_to_vector(binary_sum))
    return inputs, outputs


def convert_string_to_vector(in_string):
    vector = []
    for i in range(0, len(in_string)):
        vector.append(int(in_string[i]))
    return vector

'''func_inputs = []
func_outputs = []
num_funcs = 3
for i in range(0, num_funcs):
    func_inputs_temp, func_outputs_temp = get_random_function((-50, 50), 5, 5)
    func_inputs.append(func_inputs_temp)
    func_outputs.append(func_outputs_temp)
func_inputs = tuple(func_inputs)
func_outputs = tuple(func_outputs)'''
func_inputs, func_outputs = get_binary_add_functions(8)



print("inputs: " + str(func_inputs))
print("outputs: " + str(func_outputs))

nnet = Layers(Sigmoid, [3,25,3], Cost)
'''expected = ([.95, .05])
input = [184, 79]'''
training_cycles = 10000
learn_constant = .1
cost_threshold = .3
print(nnet)

print("initial result: " + str(nnet.get_results(func_inputs)))
print("initial cost: " + str(nnet.get_total_cost(func_outputs, nnet.get_results(func_inputs))))

stop_training = False
for i in range(0, training_cycles):
    if not stop_training:
        for j in range(0, len(func_inputs)):
            results = nnet.get_result(func_inputs[j])
            cost = nnet.get_cost(func_outputs[j], results)
            
            nnet.set_node_partials(func_outputs[j], results)
            nnet.move_across_gradient(learn_constant)
            nnet.reset()
            
        if i % (training_cycles/100) == 0:
            print(str(100 * float(i)/float(training_cycles)) + "% finished")
            tot_results = nnet.get_results(func_inputs)
            nnet.reset()
            tot_cost = nnet.get_total_cost(func_outputs, tot_results)
            print("cost: " + str(tot_cost))
            if tot_cost < cost_threshold:
                stop_training = True


for i in range(0, 7):
    print("result of " + str(i) + ": " + str(convert_string_to_vector("{0:03b}".format(i))) + str(numpy.round(numpy.asarray(nnet.get_result(convert_string_to_vector("{0:03b}".format(i)))))))

print("final result: " + str(result2))
print("final cost: " + str(nnet.get_total_cost(func_outputs, result2)))
print(nnet)
    
    
    
    