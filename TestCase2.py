import NeuronBase

inputLayer = NeuronBase.SimpleLayer(2, NeuronBase.SimpleNeuron())

middleLayer = NeuronBase.SimpleLayer(2, NeuronBase.SigmoidNeuron())

outputLayer = NeuronBase.SimpleLayer(1, NeuronBase.SigmoidNeuron())

inputLayer.connectALT(middleLayer,[[0,1],[0,1]])
middleLayer.connectALT(outputLayer,[[0],[1]])

inputLayer.setValues([2,3])

inputLayer.doCycle()
middleLayer.doCycle()
outputLayer.doCycle()

print("Test case from https://victorzhou.com/blog/intro-to-neural-networks/")
print(outputLayer.getValues())
