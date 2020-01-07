import copy
import MyMath
import random

class Dendrite:
    #This class represents a one-way connection between two neurons. It also applies a stored weight to the value passed through it.

    def __init__(self, w, n):#float, SimpleNeuron
        self.weight = w #initialises the dendrite with a weight, and a target neuron
        self.target = n

    def passValue(self, v):#float
        self.target.addInput(v*self.weight) #internal function; used for transfering values from one neuron to another

    def getWeight(self):
        return self.weight #returns the weight of the dendrite

    def setWeight(self, w):#folat
        self.weight = w #sets the weight of the dendrite


class SimpleNeuron:
    #This class describes a simple neuron, that can just add up inputs and push them through weighted outputs.
    #The reversal of the model is made to make the implementation of the backpropagation algorithm and addup procedure easier.
    #The neuron doesn't know what neurons it gets its input from. Instead it knows what neurons it has to push it's result to

    def __init__(self):
        self.value = 0.0
        self.dendrites = []
        self.cache = self.value
        self.delta = 0

    def addInput(self, inval):#float
        self.value += inval #add a value from one dendrite object; intended use: repeat for evry dendrite object connected to a particular neuron

    def resetValue(self):
        self.value = 0.0 #resets the value for another cycle

    def setValue(self, v):#float
        self.value = v #sets the value for the neuron; meant for an implementation of an input layer

    def computeValue(self):
        self.cache = self.value
        #In this particular case computeValue() does nothing. This method is provided, to allow descendant classes to modify it.
        #It's intended purpose is to modify the value of the neuron, to allow for implementing biases.

    def pushValue(self):
        x = 0 #this method passes the value through each connected dendrite
        while x < len(self.dendrites):
            self.dendrites[x].passValue(self.value)
            x += 1

    def purgeDendrites(self):
        self.dendrites = []

    def initDendrites(self, targetNeurons):#table of neurons
        x = 0 #initialisation of connections to neurons in other layers
        while x < len(targetNeurons):
            w = random.randint(0,1000)/1000.0
            self.dendrites.append(Dendrite(w,targetNeurons[x]))
            x += 1

    def initDendritesALT(self, targetNeurons, weights):#table of neurons, table of floats
        x = 0 #initialisation of connections to neurons in other layers with predefined weigths on them; intended use: debugging and loading of saved networks
        while x < len(targetNeurons):
            self.dendrites.append(Dendrite(weights[x],targetNeurons[x]))
            x += 1
    
    def derivative(self):
        return 1
