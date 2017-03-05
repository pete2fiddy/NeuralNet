import numpy

class NeuralNetTrainer:
    
    def __init__(self, nnet_in, inputs_in, expected_outputs_in):
        self.nnet = nnet_in
        self.inputs = inputs_in
        self.exp_outputs = expected_outputs_in
    
    def train(self, iterations, learn_constant, vertical_output_shift = 0, output_multiplier = 1):
        
        for i in range(0, iterations):
            results = ((numpy.asarray(self.nnet.get_results(self.inputs, vertical_output_shift = vertical_output_shift, output_multiplier = output_multiplier)) )) 
            print("initial cost: " + str(self.nnet.get_total_cost(self.exp_outputs, results)))
            self.nnet.set_multi_input_node_partials(self.exp_outputs, results)
            self.nnet.move_across_gradient(learn_constant)
            self.nnet.reset()
            if i % (iterations/20.0) == 0:
                print(str(100 * float(i)/float(iterations)) + "% finished")

        