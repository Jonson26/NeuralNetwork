import NeuronBase as NB

NN = NB.SimpleNeuralNetwork([NB.SimpleNeuron(),NB.SigmoidNeuron(),NB.LeakyReLUNeuron(0.001),"SOFTMAX"],[1,3,3,3])
#NN.connectALT([
#    [[0,0.5,1]],
#    [[1,1,1],[-1,-1,-1],[0,0,1]],
#    [[-1,-1,-1],[1,0.5,1],[-1,0,1]]])

print(NN.compute([1]))

print(NN.computeError([1,1,1]))
