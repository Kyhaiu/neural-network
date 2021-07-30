import neural_network as nn
import matrix as mat
import random


class Main():

    def __init__(self) -> None:
        pass

    def test(self):
        rna = nn.neural_network(2, 2, 1)
        training_data = [([0, 1], [1]), ([1, 0], [1]),
                         ([0, 0], [0]), ([1, 1], [0])]
        i, j = 0, 0

        while i < 10000:
            j = 0
            while j < len(training_data):
                data_learning = random.sample(training_data, 1)
                rna.train(data_learning[0][0], data_learning[0][1])
                j += 1
            i += 1

        print("guess for [0,0]:", rna.feedfoward(training_data[0][0]))
        print("guess for [0,1]:", rna.feedfoward(training_data[1][0]))
        print("guess for [1,0]:", rna.feedfoward(training_data[2][0]))
        print("guess for [1,1]:", rna.feedfoward(training_data[3][0]))

        """print('Number of nodes')
        print('\tinput: ', rna.input_nodes,
              '\thidden: ', rna.hidden_nodes,
              '\toutput', rna.output_nodes)

        print('Matrix of weights')
        print('weights_ih: ')
        print(rna.weights_ih.data)
        print('weights_ho: ')
        print(rna.weights_ho.data)
        print('Bias')
        print('Bias of Hiddens: ')
        print(rna.bias_h.data)
        print('Bias of Output: ')
        print(rna.bias_o.data)"""


main = Main()
i = 0
while i < 10:
    print("Instance ", i+1, " of RNA")
    main.test()
    i += 1


"""
rna = nn.neural_network(3, 2, 2)
print('Number of nodes')
Print('\tinput: ', rna.input_nodes,
      '\thidden: ', rna.hidden_nodes,
      '\toutput', rna.output_nodes)

print('Matrix of weights')
print('weights_ih: ')
print(rna.weights_ih.data)
print('weights_ho: ')
print(rna.weights_ho.data)
print('Bias')
print('Bias of Hiddens: ')
print(rna.bias_h.data)
print('Bias of Output: ')
print(rna.bias_o.data)

inpt = [1, 2, 3]
out = rna.feedfoward(inpt)
print("Input")
print(inpt)

print("Output:")
print(out)
"""
