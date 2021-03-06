import numpy
import random
from math import sqrt

class Node:
    
    RANDOM_BIAS_RANGE = (-1,1)#(-5,5)
    BIAS_RATE = 0.0000000000000000000000000000000001
    
    def __init__(self, activation_function_in, num_nodes_prev_layer, index):
        self.index = index
        self.dropout_rate = 0
        self.dropout = False
        self.act_func = activation_function_in
        self.init_random_weights(num_nodes_prev_layer)
        
    def init_random_weights(self, num_nodes_prev_layer):
        self.weights = numpy.zeros((num_nodes_prev_layer))
        
        for i in range(0, self.weights.shape[0]):
            std_dev = sqrt(1.0/float(num_nodes_prev_layer))
            self.weights[i] = random.gauss(0, std_dev)
            #self.weights[i] = (2*random.random()-1) 
        if num_nodes_prev_layer != 0:
            self.bias_weight = (random.random() * (Node.RANDOM_BIAS_RANGE[1]-Node.RANDOM_BIAS_RANGE[0])) + Node.RANDOM_BIAS_RANGE[0]
        else:
            self.bias_weight = None
        
    def set_output(self, num = None):
        if num == None:
            if not self.dropout:
                if self.bias_weight != None:
                    self.output = self.act_func.func(self.sum + self.bias_weight) 
                else:
                    self.output = self.act_func.func(self.sum) 
            else:
                self.output = 0
        else:
            self.output = num
    
    
    
    def set_dropout(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.dropout = (random.random() < dropout_rate)
    
    def get_deriv_at_output(self):
        return self.act_func.dfunc(self.output)
    
    def get_output(self):
        return self.output
    
    def set_sum(self, sum_in):
        self.sum = sum_in
    
    def get_sum(self):
        return self.sum
    
    def get_weighted_sum(self, prev_outputs):
        return numpy.dot(self.weights * (1.0/(1.0-self.dropout_rate)), prev_outputs)
    
    '''def get_weight_partials(self, cost_partials, next_layer):
        partials = numpy.zeros((self.weights.shape[0]))
        for i in range(0, partials.shape[0]):
    '''   
    '''
    def set_weight_partials(self, cost_partials, next_layer):
        self.weight_partials = numpy.zeros((self.weights.shape[0]))
        if next_layer == None:
            return self.get_deriv_at_output() * cost_partials[self.index]
        '''
    
    def set_weight_partials(self, prev_layer, next_layer, cost_partials):
        self.partials = numpy.zeros((self.weights.shape[0]))
        if not self.dropout:
            if next_layer == None:
                for i in range(0, self.partials.shape[0]):
                    self.partials[i] = prev_layer[i].get_output()*self.get_deriv_at_output()*cost_partials[self.index]
            else:
                for i in range(0, self.partials.shape[0]):
                    self.partials[i] = prev_layer[i].get_output()*self.get_deriv_at_output()#change in neuron output/change in weight
                    sum = 0
                    for j in range(0, len(next_layer)):
                        sum += next_layer[j].get_partials()[self.index]
                    self.partials[i] *= sum
        
        
        
        
    def move_across_gradient(self, learn_constant):
        if not self.dropout:
            self.weights = numpy.add(self.weights, numpy.asarray(self.partials) * learn_constant)
            '''for i in range(0, self.weights.shape[0]):
                self.weights[i] -= self.partials[i] * learn_constant'''
            if self.bias_weight != None:
                self.bias_weight += self.get_deriv_at_output() * Node.BIAS_RATE
    
        
    def reset_sums_outputs_and_partials(self):
        self.sum = 0
        self.output = 0
        self.partials = numpy.zeros((self.partials.shape[0]))
        self.dropout_rate = 0
        self.dropout = False
                  
    def get_partials(self):
        return self.partials              
             
      
    
    def get_weight_partials(self):
        return self.weight_partials
    
    def __repr__(self):
        return "Node with weights: " + str(self.weights) + " and bias weight: " + str(self.bias_weight)