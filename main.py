import neural_network as nn
import matrix as mat


class Main():

    def __init__(self) -> None:
        pass

    def test(self):
        matrix = mat.matrix(2, 3)
        print(matrix.data)
        matrix.data = matrix.randomize(matrix.data)
        print(matrix.data)


main = Main()
main.test()
