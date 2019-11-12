import copy
import MyMath

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

    def addInput(self, inval):#float
        self.value += inval #add a value from one dendrite object; intended use: repeat for evry dendrite object connected to a particular neuron

    def resetValue(self):
        self.value = 0.0 #resets the value for another cycle

    def setValue(self, v):#float
        self.value = v #sets the value for the neuron; meant for an implementation of an input layer

    def computeValue(self):
        None
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
            self.dendrites.append(Dendrite(1,targetNeurons[x]))
            x += 1

    def initDendritesALT(self, targetNeurons, weights):#table of neurons, table of floats
        x = 0 #initialisation of connections to neurons in other layers with predefined weigths on them; intended use: debugging and loading of saved networks
        while x < len(targetNeurons):
            self.dendrites.append(Dendrite(weights[x],targetNeurons[x]))
            x += 1

class SigmoidNeuron(SimpleNeuron):
    #implementation of a sigmoid neuron
    
    def computeValue(self):#overwritten activation function
        self.value = MyMath.sigmoid(self.value)
    

class ReLUNeuron(SimpleNeuron):
    #implementation of a rectified linear unit neuron
    def computeValue(self):#overwritten activation function
        self.value = max(0, self.value)

class LeakyReLUNeuron(SimpleNeuron):
    #implementation of a leaky rectified linear unit neuron
    def __init__(self,parameter):#float (should be something near zero)
        self.parameter = parameter #the activation function of this type of neuron uses a parameter, that need to be initialised
        super().__init__()

    def computeValue(self):#overwritten activation function
        if self.value < 0:
            self.value = self.value * self.parameter

class SoftmaxNeuron(SimpleNeuron):
    def __init__(self):
        self.valuesIn = []
        self.valuesOut = []
        self.value = self.valuesOut
        self.dendrites = []

    def addInput(self, inval):
        self.valuesIn.append(inval)

    def computeValue():
        e = MyMath.exponent(1,100)
        E = 0
        for x in self.valuesIn:
            E += e**x
        for x in self.valuesIn:
            self.valuesOut.append((e**x)/E)

class SimpleLayer:
    #abstraction class, meant to facilitate the construction of layers for the user (and to hide some of the inner workings)
    def __init__(self, amountOfNeurons, neuronTemplate):#int, some neuron object
        self.neurons = []#since there can be different types of neurons, the layer is constructed from a neuron template, which is then cloned for each neuron in the layer. This allows for an unlimited amount of custom neurons, and also accepts user-defined neurons, as long as they are child classes to the SimpleNeuron class.
        i = 0
        while i < amountOfNeurons:
            self.neurons.append(copy.deepcopy(neuronTemplate))
            i += 1

    def purgeDendritres(self):
        for x in self.neurons:
            x.purgeConnections()

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
        
class SimpleNeuralNetwork:

    def __init__(self, tableOfLayerTypes, tableOfLayerCounts, passResultThroughSoftmax):#table of Neuron objects; table of ints; boolean
        i = 0
        self.PRTSM = passResultThoughSoftmax
        self.layers = []
        while i < len(tableOfLayerCounts):
            self.layers.append(SimpleLayer(tableOfLayerCounts[i], tableOfLayerTypes[i]))
            i += 1
        if self.PRTSM:
            self.layers.append(SimpleLayer(1,SoftmaxNeuron()))
        i = 0
        while i < (len(self.layers)-1):
            self.layers[i].connect(self.layers[i+1])
            i += 1

    def connectALT(self, weightsCubed):
        for x in self.layers:
            x.purgeConnections()
        i = 0
        while i < (len(self.layers)-1):
            self.layers[i].connectALT(self.layers[i+1], weightsCubed[i])
            i += 1
        
    def compute(self, inputTable):
        self.layers[0].setValues(inputTable)
        for x in self.layers:
            x.doCycle()
        if self.PRTSM:
            return self.layers[len(self.layers)-1].getValues[0]
        else:
            return self.layers[len(self.layers)-1].getValues[0]
