import NeuronBase

inputLayer = NeuronBase.SimpleLayer(2, NeuronBase.SimpleNeuron())

middleLayer = NeuronBase.SimpleLayer(3, NeuronBase.SigmoidNeuron())

outputLayer = NeuronBase.SimpleLayer(1, NeuronBase.SigmoidNeuron())

inputLayer.connectALT(middleLayer,[[0.2,0.6,0.1],[0.8,0.3,0.7]])
middleLayer.connectALT(outputLayer,[[0.4,0.5,0.9]])

inputLayer.setValues([2,9])

inputLayer.doCycle()
middleLayer.doCycle()
outputLayer.doCycle()

print("Test case from https://enlight.nyc/projects/neural-network/")
print(outputLayer.getValues())
