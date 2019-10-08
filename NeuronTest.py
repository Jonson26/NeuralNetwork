import NeuronBase

inputLayer = []

middleLayerA = []

middleLayerB = []

outputLayer = []

inputLayer.append(NeuronBase.SimpleNeuron())

i = 0
while i < 3:
    middleLayerA.append(NeuronBase.SimpleNeuron())
    i += 1

i = 0
while i < 3:
    middleLayerB.append(NeuronBase.SimpleNeuron())
    i += 1

outputLayer.append(NeuronBase.SimpleNeuron())

for x in inputLayer:
    x.initDendrites(middleLayerA)

for x in middleLayerA:
    x.initDendrites(middleLayerB)

for x in middleLayerB:
    x.initDendrites(outputLayer)

for x in inputLayer:
    x.resetValue()
    x.setValue(1)
    x.computeValue()
    x.pushValue() 

for x in middleLayerA:
    x.computeValue()
    x.pushValue() 

for x in middleLayerB:
    x.computeValue()
    x.pushValue()

print(outputLayer[0].value)
