class Dendrite:
    #This class represents a one-way connection between two neurons. It also applies a stored weight to the value passed through it.

    def __init__(self, w, n):#float, SimpleNeuron
        self.weight = w
        self.target = n

    def passValue(self, v):#float
        self.target.addInput(v*self.weight)

    def getWeight(self):
        return self.weight

    def setWeight(self, w):#folat
        self.weight = w


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
        self.value = 0.0

    def setValue(self, v):#float
        self.value = v

    def computeValue(self):
        None
        #In this particular case computeValue() does nothing. This method is provided, to allow descendant classes to modify it.
        #It's intended purpose is to modify the value of the neuron, to allow for implementing biases.

    def pushValue(self):
        x = 0
        while x < len(self.dendrites):
            self.dendrites[x].passValue(self.value)
            x += 1

    def initDendrites(self, targetNeurons):
        x = 0
        while x < len(targetNeurons):
            self.dendrites.append(Dendrite(1,targetNeurons[x]))
            x += 1

