import typing
import matrix as mat


class NeuralNetwork:

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """
            Construtor da classe NeuralNetwork

            ---
            Parametros:
                * input_nodes: Integer - Numero de neurônios de entrada.
                * hidden_nodes: Integer - Numero de neurônios da camada oculta.
                * output_nodes: Integer - Numero de neurônios da camada de saída.
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = mat.matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = mat.matrix(self.output_nodes, self.hidden_nodes)

        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = mat.matrix(self.hidden_nodes, 1)
        self.bias_o = mat.matrix(self.output_nodes, 1)

    def __setattr__(self, name: str, value: typing.Any) -> None:
        """
            Definição de atributos da classe NeuralNetwork

            ---
            Atributos:
                * input_nodes: Integer - Numero de neurônios de entrada.
                * hidden_nodes: Integer - Numero de neurônios da camada oculta.
                * output_nodes: Integer - Numero de neurônios da camada de saída.

            ---
        """
        if name == 'input_nodes':
            super(NeuralNetwork, self).__setattr__(name, value)
        if name == 'hidden_nodes':
            super(NeuralNetwork, self).__setattr__(name, value)
        if name == 'output_nodes':
            super(NeuralNetwork, self).__setattr__(name, value)
        if name == 'weights_ih':
            super(NeuralNetwork, self).__setattr__(name, value)
        if name == 'weights_ho':
            super(NeuralNetwork, self).__setattr__(name, value)
        if name == 'bias_h':
            super(NeuralNetwork, self).__setattr__(name, value)
        if name == 'bias_o':
            super(NeuralNetwork, self).__setattr__(name, value)

    def feedfoward(self, input_array):

        input = mat.matrix.from_array(input_array)

        hidden = mat.matrix.mat_mul(self.weights_ih, input)

        hidden.add_scale(self.bias_h)
