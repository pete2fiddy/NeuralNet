from NNet.Main.Layers import Layers
import NNet.Function.Sigmoid as Sigmoid
import NNet.Function.Cost as Cost
import numpy
import random

def get_random_function(dim_input, input_range, dim_output):
    inputs = tuple([input_range[0]+random.random()*(input_range[1]-input_range[0]) for i in range(0, dim_input)])
    outputs = tuple([random.random() for i in range(0, dim_output)])
    inputs_and_outputs = tuple((inputs, outputs))
    
    return inputs_and_outputs

num_funcs = 3
funcs = []
dim_input = 3
dim_output = 3
input_range = (-50, 50)
for i in range(0, num_funcs):
    funcs.append(get_random_function(dim_input, input_range, dim_output))
    
nnet = Layers(Sigmoid, [3,3,3], Cost)
expected = []

print("function table: " + str(funcs))

for i in range(0, len(funcs)):
    expected.append(funcs[i][1])
print("expected: " + str(expected))
'''expected = ([.9999, .0001, .354, .78621, .77])
input = [184, 79, 48, -100, 95]'''
training_cycles = 10000
learn_constant = .1
print(nnet)

print("initial result: " + str(nnet.get_result(funcs)))
print("initial cost: " + str(nnet.get_cost(expected, nnet.get_result(funcs))))


for i in range(0, training_cycles):
    result = nnet.get_result(funcs)
    cost = nnet.get_cost(expected, result)
    nnet.set_node_partials(expected, result)
    nnet.move_across_gradient(learn_constant)
    nnet.reset()
    result2 = nnet.get_result(funcs)
    if i % (training_cycles/20) == 0:
        print(str(float(float(i)/float(training_cycles)) * 100) + "% finished")

print("final result: " + str(result2))
print("final cost: " + str(nnet.get_cost(expected, result2)))
    
'''result = nnet.get_result([1,1])
print("result: " + str(result))
print("cost: " + str(nnet.get_cost(expected, result)))
nnet.set_node_partials(expected, result)
nnet.move_across_gradient(-.05)
nnet.reset()

result2 = nnet.get_result([1,1])
print("second result: " + str(result2))
print("cost: " + str(nnet.get_cost(expected, result2)))'''