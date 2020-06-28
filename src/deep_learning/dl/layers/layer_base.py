class LayerBase:
    def __init__(self):
        pass
    
    def initialize_parameters(self):
        pass

    def parameters(self):
        return {}
    
    def gradients(self):
        return {}

    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, x):
        raise NotImplementedError()
