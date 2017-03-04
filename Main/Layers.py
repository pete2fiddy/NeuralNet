from NNet.Main.Layer import Layer
import numpy
class Layers:
    
    def __init__(self, act_func, layer_dims, cost_func_in):
        self.init_layers(act_func, layer_dims)
        self.cost_func = cost_func_in
        
    def init_layers(self, act_func, layer_dims):
        self.layers = []
        
        self.input_layer = Layer(act_func, layer_dims[0], None)
        self.layers.append(self.input_layer)
        for i in range(1, len(layer_dims)):
            self.layers.append(Layer(act_func, layer_dims[i], self.layers[i-1]))
    
    def set_input(self, inputs):
        for i in range(0, len(self.input_layer)):
            self.input_layer[i].set_sum(inputs[i])
            self.input_layer[i].set_output()
    
    def get_result(self, inputs):
        outputs = []
        for input_index in range(0, len(inputs)):
            self.set_input(inputs[input_index][0])
            for i in range(1, len(self.layers)):
                self.layers[i].set_node_sums()
                self.layers[i].set_node_outputs()
            outputs.append(self.layers[len(self.layers)-1].get_layer_outputs())
        return tuple(outputs)
            
        #return self.layers[len(self.layers)-1].get_layer_outputs()
    
    
    def set_node_partials(self, expected, result):
        i = len(self.layers) - 2
        cost_partials = self.cost_func.get_partials(expected, result)
        self.layers[len(self.layers) - 1].set_node_partials(None, cost_partials)
        while i > 0:
            self.layers[i].set_node_partials(self.layers[i+1], cost_partials)
            i -= 1
    
    def move_across_gradient(self, learn_constant):
        for i in range(0, len(self.layers)):
            self.layers[i].move_across_gradient(learn_constant)
    
    def get_cost(self, expected, result):
        return self.cost_func.get_cost(expected, result)
           
    def reset(self):        
        for i in range(1, len(self.layers)):
            self.layers[i].reset_nodes()
    
    def __repr__(self):
        out = ""
        for i in range(0, len(self.layers)):
            out += "layer " + str(i) + ": " + str(self.layers[i]) + "\n"
        return out