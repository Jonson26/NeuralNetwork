import copy
import MyMath
import random
from NeuronBase import *

class SigmoidNeuron(SimpleNeuron):
    #implementation of a sigmoid neuron
    
    def computeValue(self):#overwritten activation function
        self.cache = self.value
        self.value = MyMath.sigmoid(self.value)

    def derivative():
        return self.cache*(1-self.cache)
    

class ReLUNeuron(SimpleNeuron):
    #implementation of a rectified linear unit neuron
    def computeValue(self):#overwritten activation function
        self.cache = self.value
        self.value = max(0, self.value)

    def derivative(self):
        if self.cache > 0:
            return 1
        return 0

class LeakyReLUNeuron(SimpleNeuron):
    #implementation of a leaky rectified linear unit neuron
    def __init__(self,parameter):#float (should be something near zero)
        self.parameter = parameter #the activation function of this type of neuron uses a parameter, that need to be initialised
        super().__init__()

    def computeValue(self):#overwritten activation function
        self.cache = self.value
        if self.value < 0:
            self.value = self.value * self.parameter

    def derivative(self):
        if self.cache > 0:
            return 1
        return self.parameter
