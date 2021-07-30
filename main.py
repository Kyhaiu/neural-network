import neural_network as nn
import matrix as mat


class Main():

    def __init__(self) -> None:
        pass

    def test(self):
        rna = nn.neural_network(3, 2, 2)
        print('Number of nodes')
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
        print(rna.bias_o.data)

        inpt = [1, 2, 3]
        out = rna.feedfoward(inpt)
        print("Input")
        print(inpt)

        print("Output:")
        print(out)


main = Main()
main.test()
