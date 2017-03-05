

class PCAClassifier:
    
    def __init__(self, named_projections_in, nnet_in):
        self.nnet = nnet_in
        self.named_projections = named_projections_in
        
        