from NNet.Main.Layers import Layers
import NNet.Function.Sigmoid as Sigmoid
import NNet.Function.Cost as Cost
import random
from NNet.Main.NeuralNetTrainer import NeuralNetTrainer
import EigenFit.Load.NumpyLoader as NumpyLoader
from EigenFit.DataMine.Categorizer import Categorizer
from DataMine.KMeansCompare import KMeansCompare
import numpy
import EigenFit.Vector.VectorMath as VectorMath

def get_random_function(input_range, dim_input, dim_output):
    inputs = tuple([input_range[0] + random.random()*(input_range[1]-input_range[0]) for i in range(0, dim_input)])
    outputs = tuple([random.random() for i in range(0, dim_output)])
    return inputs, outputs


def get_letter_closest_to_vector(vector, named_projections):
    output = sorted( named_projections, key = lambda named_projection: numpy.linalg.norm( numpy.subtract(vector, named_projection.get_unit_mean()) ) )
    print("output: " + str(output))
    return output[0].get_name()


'''

func_inputs = []
func_outputs = []
num_funcs = 3
for i in range(0, num_funcs):
    func_inputs_temp, func_outputs_temp = get_random_function((-50, 50), 5, 5)
    func_inputs.append(func_inputs_temp)
    func_outputs.append(func_outputs_temp)
func_inputs = tuple(func_inputs)
func_outputs = tuple(func_outputs)

print("inputs: " + str(func_inputs))
print("outputs: " + str(func_outputs))
'''

num_dims = 20
nnet = Layers(Sigmoid, [num_dims, num_dims, num_dims], Cost)


training_cycles = 100
learn_constant = .0001
#print(nnet)

base_path = "/Users/phusisian/Desktop/Senior year/SUAS/Competition Files/NEWLETTERPCA"
eigenvectors = NumpyLoader.load_numpy_arr(base_path + "/Data/Eigenvectors/eigenvectors 0.npy")
projections_path = base_path + "/Data/Projections"
mean = NumpyLoader.load_numpy_arr(base_path + "/Data/Mean/mean_img 0.npy")
letter_categorizer = Categorizer(eigenvectors, mean, projections_path, KMeansCompare, num_dims)
named_projections = letter_categorizer.get_named_projections()

inputs = []
outputs = []

for i in range(0, len(named_projections)):
    iterinputs, iteroutputs = named_projections[i].as_set()
    for i in range(0, len(iterinputs)):
        inputs.append(iterinputs[i])
        outputs.append(iteroutputs[i])

nn_trainer = NeuralNetTrainer(nnet, inputs, outputs)
out_multiplier = 1#2.0
out_shift = 0#-.5
nn_trainer.train(training_cycles, learn_constant, vertical_output_shift = out_shift, output_multiplier = out_multiplier)

num_wrong = 0
num_total = 0
for i in range(0, len(named_projections)):
    inputs, outputs = named_projections[i].as_set()
    for named_proj_index in range(0, named_projections[i].get_projections().shape[0]):
        #print("named projection: " + str(named_projections[i]))
        iter_result = (nnet.get_result((inputs[named_proj_index])) + out_shift) * out_multiplier#nnet.get_result((VectorMath.unit_vector(numpy.asarray(named_projections[i].get_projections()[named_proj_index])) + out_shift)*out_multiplier)
        print("distance between output and correct answer" + str(numpy.linalg.norm( numpy.subtract(iter_result, outputs[named_proj_index]) ) ))
        print("cost: " + str(Cost.get_cost(iter_result, outputs[named_proj_index])))
        #print("iter result: " + str(iter_result))
        '''letter_closest_to_result = get_letter_closest_to_vector(iter_result, named_projections)
        if letter_closest_to_result != named_projections[i].get_name():
            num_wrong += 1
            #print("got one wrong :(")
        #else:
            #print("got one correct!")
        '''
        num_total += 1
        
print("total error: " + str(float(num_wrong)/float(num_total)))


'''
print("initial result: " + str(nnet.get_results(func_inputs)))
print("initial cost: " + str(nnet.get_total_cost(func_outputs, nnet.get_results(func_inputs))))

for i in range(0, training_cycles):
    results = nnet.get_results(func_inputs)
    cost = nnet.get_total_cost(func_outputs, results)
    nnet.set_multi_input_node_partials(func_outputs, results)
    nnet.move_across_gradient(learn_constant)
    nnet.reset()
    if i % (training_cycles/20) == 0:
        print(str(100 * float(i)/float(training_cycles)) + "% finished")

print("final result: " + str(result2))
print("final cost: " + str(nnet.get_total_cost(func_outputs, result2)))
print(nnet)
    
'''
    
    