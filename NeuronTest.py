import NeuronBase

outputNeuron = NeuronBase.SimpleNeuron()

middleNeuron = NeuronBase.SimpleNeuron()
middleNeuron.dendrites = [NeuronBase.Dendrite(1/2, outputNeuron)]

inputNeuron = NeuronBase.SimpleNeuron()
inputNeuron.dendrites = [NeuronBase.Dendrite(7/10, middleNeuron)]

inputNeuron.setValue(1)
inputNeuron.computeValue()
inputNeuron.pushValue()

middleNeuron.computeValue()
middleNeuron.pushValue()

outputNeuron.computeValue()
print(outputNeuron.value)
