import NeuronFramework as NF

NN = NF.SimpleNeuralNetwork([NF.SimpleNeuron(),NF.SigmoidNeuron(),NF.LeakyReLUNeuron(0.001),"SOFTMAX"],[1,3,3,3])
#NN.connectALT([
#    [[0,0.5,1]],
#    [[1,1,1],[-1,-1,-1],[0,0,1]],
#    [[-1,-1,-1],[1,0.5,1],[-1,0,1]]])

print(NN.compute([1]))

print(NN.computeError([1,1,1]))
