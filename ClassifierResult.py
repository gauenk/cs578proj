import numpy as np

class ClassifierResult:
    def __init__(self, name, zo_loss, test_loss):
        self.name = name
        self.test_name = name + ' Test'
        self.zero_one_loss = zo_loss
        self.test_loss = test_loss

class SVMClassifierResult(ClassifierResult):
    def __init__(self, name, zo_loss, test_loss, params):
        ClassifierResult.__init__(self, name, test_loss, zo_loss)
        self.params = params

class GSSVMClassifierResult(SVMClassifierResult):
    def __init__(self, name, zo_loss, test_loss, params):
        ClassifierResult.__init__(self, name, test_loss, zo_loss)
        self.params = params
        self.columns = []