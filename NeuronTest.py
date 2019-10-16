import NeuronBase

inputLayer = NeuronBase.SimpleLayer(1, NeuronBase.SimpleNeuron())

middleLayerA = NeuronBase.SimpleLayer(3, NeuronBase.SimpleNeuron())

middleLayerB = NeuronBase.SimpleLayer(3, NeuronBase.LeakyReLUNeuron(0.0001))

outputLayer = NeuronBase.SimpleLayer(1, NeuronBase.SimpleNeuron())

inputLayer.connectALT(middleLayerA,[[0,0.5,1]])
middleLayerA.connectALT(middleLayerB,[[1,1,1],[-1,-1,-1],[0,0,1]])
middleLayerB.connect(outputLayer)

inputLayer.setValues([1])

inputLayer.doCycle()
print(inputLayer.getValues())
print([[0,0.5,1]])
middleLayerA.doCycle()
print(middleLayerA.getValues())
print([[1,1,1],[-1,-1,-1],[0,0,1]])
middleLayerB.doCycle()
print(middleLayerB.getValues())
print([[1],[1],[1]])
outputLayer.doCycle()

print(outputLayer.getValues())
