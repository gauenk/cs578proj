import numpy as np

class ClassifierResult:
    def __init__(self, name, zo_loss):
        self.name = name
        self.zero_one_loss = zo_loss

class SVMClassifierResult(ClassifierResult):
    def __init__(self, name, zo_loss, params):
        ClassifierResult.__init__(self, name, zo_loss)
        self.params = params
