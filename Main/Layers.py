from NNet.Main.Layer import Layer
import numpy
class Layers:
    
    def __init__(self, act_func, layer_dims, cost_func_in):
        self.init_layers(act_func, layer_dims)
        self.cost_func = cost_func_in
        
    def init_layers(self, act_func, layer_dims):
        self.layers = []
        
        self.input_layer = Layer(act_func, layer_dims[0], None)
        #self.layers.append(self.input_layer)
        second_layer = Layer(act_func, layer_dims[0], self.input_layer)
        self.layers.append(second_layer)
        for i in range(1, len(layer_dims)):
            self.layers.append(Layer(act_func, layer_dims[i], self.layers[i-1]))
        print("len layers: " + str(len(self.layers)))
    
    def set_input(self, inputs):
        for i in range(0, len(self.input_layer)):
            #self.input_layer[i].set_sum(inputs[i])
            self.input_layer[i].set_output(inputs[i])
    
    def get_result(self, inputs):
        self.set_input(inputs)
        for i in range(0, len(self.layers)):
            self.layers[i].set_node_sums()
            self.layers[i].set_node_outputs()
        return self.layers[len(self.layers)-1].get_layer_outputs()
    
    def get_results(self, mult_inputs):
        outputs = []
        for i in range(0, len(mult_inputs)):
            outputs.append(self.get_result(mult_inputs[i]))
        return tuple(outputs)
    
    def set_node_partials(self, expected, result):
        i = len(self.layers) - 2
        cost_partials = self.cost_func.get_partials(expected, result)
        self.layers[len(self.layers) - 1].set_node_partials(None, cost_partials)
        while i >= 0:
            self.layers[i].set_node_partials(self.layers[i+1], cost_partials)
            i -= 1
    
           
    def set_multi_input_node_partials(self, expecteds, results):
        i = len(self.layers) - 2
        cost_partials = self.cost_func.get_total_partials(expecteds, results)
        self.layers[len(self.layers) - 1].set_node_partials(None, cost_partials)
        while i >= 0:
            self.layers[i].set_node_partials(self.layers[i+1], cost_partials)
            i -= 1
    
    def move_across_gradient(self, learn_constant):
        for i in range(0, len(self.layers)):
            self.layers[i].move_across_gradient(learn_constant)
    
    def get_cost(self, expected, result):
        return self.cost_func.get_cost(expected, result)
    
    def get_total_cost(self, expecteds, results):
        return self.cost_func.get_total_cost(expecteds, results)
    
    def reset(self):   
             
        for i in range(0, len(self.layers)):
            self.layers[i].reset_nodes()
    
    def __repr__(self):
        out = ""
        for i in range(0, len(self.layers)):
            out += "layer " + str(i) + ": " + str(self.layers[i]) + "\n"
        return out