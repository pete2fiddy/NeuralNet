from NNet.Main.Layers import Layers
import NNet.Function.Sigmoid as Sigmoid
import NNet.Function.Cost as Cost

nnet = Layers(Sigmoid, [2,4,2], Cost)
expected = ([.95, .05])
input = [184, 79]
training_cycles = 10000
learn_constant = .1
print(nnet)

print("initial result: " + str(nnet.get_result(input)))
print("initial cost: " + str(nnet.get_cost(expected, nnet.get_result(input))))


for i in range(0, training_cycles):
    result = nnet.get_result(input)
    cost = nnet.get_cost(expected, result)
    nnet.set_node_partials(expected, result)
    nnet.move_across_gradient(learn_constant)
    nnet.reset()
    result2 = nnet.get_result(input)

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