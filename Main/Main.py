from NNet.Main.Layers import Layers
import NNet.Function.Sigmoid as Sigmoid
import NNet.Function.Cost as Cost
import random



def get_random_function(input_range, dim_input, dim_output):
    inputs = tuple([input_range[0] + random.random()*(input_range[1]-input_range[0]) for i in range(0, dim_input)])
    outputs = tuple([random.random() for i in range(0, dim_output)])
    return inputs, outputs





func_inputs = []
func_outputs = []
num_funcs = 3
for i in range(0, num_funcs):
    func_inputs_temp, func_outputs_temp = get_random_function((-50, 50), 3, 3)
    func_inputs.append(func_inputs_temp)
    func_outputs.append(func_outputs_temp)
func_inputs = tuple(func_inputs)
func_outputs = tuple(func_outputs)

print("inputs: " + str(func_inputs))
print("outputs: " + str(func_outputs))

nnet = Layers(Sigmoid, [3,3,3], Cost)
'''expected = ([.95, .05])
input = [184, 79]'''
training_cycles = 100000
learn_constant = .05
print(nnet)

print("initial result: " + str(nnet.get_results(func_inputs)))
print("initial cost: " + str(nnet.get_total_cost(func_outputs, nnet.get_results(func_inputs))))

for i in range(0, training_cycles):
    results = nnet.get_results(func_inputs)
    cost = nnet.get_total_cost(func_outputs, results)
    nnet.set_multi_input_node_partials(func_outputs, results)
    nnet.move_across_gradient(learn_constant)
    nnet.reset()
    result2 = nnet.get_results(func_inputs)
    if i % (training_cycles/20) == 0:
        print(str(100 * float(i)/float(training_cycles)) + "% finished")

print("final result: " + str(result2))
print("final cost: " + str(nnet.get_total_cost(func_outputs, result2)))
    
    
    
    
    