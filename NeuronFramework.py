import copy
import MyMath
import random
from NeuronBase import *
from NeuronExtension import *

class SimpleLayer:
    #abstraction class, meant to facilitate the construction of layers for the user (and to hide some of the inner workings)
    def __init__(self, amountOfNeurons, neuronTemplate):#int, some neuron object
        self.neurons = []#since there can be different types of neurons, the layer is constructed from a neuron template, which is then cloned for each neuron in the layer. This allows for an unlimited amount of custom neurons, and also accepts user-defined neurons, as long as they are child classes to the SimpleNeuron class.
        self.delta = 0
        i = 0
        while i < amountOfNeurons:
            self.neurons.append(copy.deepcopy(neuronTemplate))
            i += 1

    def purgeDendrites(self):
        for x in self.neurons:
            x.dendrites = []

    def connect(self, nextLayer):#SimpleLayer
        for x in self.neurons:#sets up the connections between the neurons in this layer, and the next one
            x.initDendrites(nextLayer.neurons)

    def connectALT(self, nextLayer, weightsSquared):#SimpleLayer, table of tables of floats
        i=0 #setup of connections between the neurons this layer, and the next one with predefined weigths on them; intended use: debugging and loading of saved networks
        for x in weightsSquared:
            self.neurons[i].initDendritesALT(nextLayer.neurons, x)
            i += 1

    def doCycle(self):
        for x in self.neurons:#executes the activation function and pushes the result to the next layer
            x.computeValue()
            x.pushValue()

    def setValues(self, values):#table of floats
        i = 0#sets the value for the neurons in this layer; meant for an implementation of an input layer
        while i < len(self.neurons):
            self.neurons[i].setValue(values[i])
            i += 1

    def getValues(self):
        out=[]#returns al the values of the neurons in the layer as a table; this is essentially, where one would get the output in the output layer, but is also useful for debugging
        for x in self.neurons:
            out.append(x.value)
        return out

class SoftMaxLayer(SimpleLayer):

    def doCycle(self):
        e = MyMath.exponent(1,100)
        n = []
        E = 0
        for x in self.neurons:
            E += e**x.value
        for x in self.neurons:
            x.value = (e**x.value)/E
            x.pushValue()
    
class SimpleNeuralNetwork:

    def __init__(self, tableOfLayerTypes, tableOfLayerCounts):#table of Neuron objects or "SOFTMAX" strings; table of ints
        i = 0
        self.layers = []
        while i < len(tableOfLayerCounts):
            if type(tableOfLayerTypes[i])is str:
                if tableOfLayerTypes[i]=="SOFTMAX":
                    self.layers.append(SoftMaxLayer(tableOfLayerCounts[i], SimpleNeuron()))
                else:
                    print("Ya messed up, idiot. There is an illegeal input in the layertypes but i catched it")
            else:
                self.layers.append(SimpleLayer(tableOfLayerCounts[i], tableOfLayerTypes[i]))
            i += 1

        i = 0
        while i < (len(self.layers)-1):
            self.layers[i].connect(self.layers[i+1])
            i += 1

    def connectALT(self, weightsCubed):
        for x in self.layers:
            x.purgeDendrites()
        i = 0
        while i < (len(self.layers)-1):
            self.layers[i].connectALT(self.layers[i+1], weightsCubed[i])
            i += 1
        
    def compute(self, inputTable):
        self.layers[0].setValues(inputTable)
        for x in self.layers:
            x.doCycle()
        return self.layers[len(self.layers)-1].getValues()

    def computeError(self, targetTable):
        E = []
        O = self.layers[len(self.layers)-1].getValues()
        i = 0
        while(i<len(O)):
            E.append((O[i] - targetTable[i])**2)
            i += 1
        return E
