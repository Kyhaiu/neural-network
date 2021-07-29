import neural_network as nn


class Main():

    def __init__(self) -> None:
        pass

    def test(self):
        rna = nn.NeuralNetwork()
        rna.neural_network(3, 2, 1)
        matrix = rna.matrix(2, 2)
        print(matrix)
        matrix = rna.add_scale(matrix)
        print(matrix)


main = Main()
main.test()
