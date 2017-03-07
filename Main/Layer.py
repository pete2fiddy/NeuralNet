from NNet.Main.Node import Node
import numpy

class Layer:
    
    def __init__(self, act_func_in, num_nodes_in, prev_layer_in):
        self.prev_layer = prev_layer_in
        self.init_nodes(act_func_in, num_nodes_in)
       
            
    def init_nodes(self, act_func, num_nodes):
        if self.prev_layer != None:
            self.nodes = [Node(act_func, len(self.prev_layer), i) for i in range(0, num_nodes)]
        else:
            self.nodes = [Node(act_func, 0, i) for i in range(0, num_nodes)]
        
        self.nodes = numpy.asarray(self.nodes, dtype = object)
            
    def set_node_sums(self):
        prev_layer_outputs = self.prev_layer.get_layer_outputs()
        for i in range(0, len(self.nodes)):
            self.nodes[i].set_sum(self.nodes[i].get_weighted_sum(prev_layer_outputs))
    
    def set_node_outputs(self):
        for i in range(0, len(self)):
            self[i].set_output()
    
    def set_node_partials(self, next_layer, cost_partials):
        for i in range(0, len(self)):
            self[i].set_weight_partials(self.prev_layer, next_layer, cost_partials)
    
    def move_across_gradient(self, learn_constant):
        for i in range(0, len(self)):
            self[i].move_across_gradient(learn_constant)
    
    def reset_nodes(self):
        for i in range(0, len(self)):
            self[i].reset_sums_outputs_and_partials()
    
    def set_dropout_nodes(self, dropout_rate):
        for i in range(0, len(self)):
            self[i].set_dropout(dropout_rate)
    
    def get_layer_outputs(self):
        return numpy.asarray([self[i].get_output() for i in range(0, len(self))])#numpy.asarray([self.prev_layer[i].get_output() for i in range(0, len(self.prev_layer))])
    
    def get_cost(self):
        return self.cost_func.get_cost()
    
    def __len__(self):
        return self.nodes.shape[0]
    
    def __getitem__(self, index):
        return self.nodes[index]
    
    def __repr__(self):
        out = ""
        for i in range(0, len(self)):
            out += "Node " + str(i) + ": " + str(self[i]) + ", "
        return out