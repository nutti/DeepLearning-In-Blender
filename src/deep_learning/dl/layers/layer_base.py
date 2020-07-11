class LayerBase:

    name_counter = {}

    def __init__(self):
        cls = self.__class__

        if self.id() not in cls.name_counter:
            cls.name_counter[self.id()] = 0

        self.name_ = "{}_{}".format(self.id(), cls.name_counter[self.id()])
        cls.name_counter[self.id()] += 1


    def id(self):
        raise NotImplementedError()

    def name(self):
        return self.name_
    
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
