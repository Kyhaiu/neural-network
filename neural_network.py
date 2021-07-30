import typing
import math
import matrix as mat


class neural_network:

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

        self.bias_h.randomize()
        self.bias_o.randomize()

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
            super(neural_network, self).__setattr__(name, value)
        if name == 'hidden_nodes':
            super(neural_network, self).__setattr__(name, value)
        if name == 'output_nodes':
            super(neural_network, self).__setattr__(name, value)
        if name == 'weights_ih':
            super(neural_network, self).__setattr__(name, value)
        if name == 'weights_ho':
            super(neural_network, self).__setattr__(name, value)
        if name == 'bias_h':
            super(neural_network, self).__setattr__(name, value)
        if name == 'bias_o':
            super(neural_network, self).__setattr__(name, value)

    @staticmethod
    def sigmoid(x) -> float:
        return (1/math.exp(-x))

    def feedfoward(self, input_array: typing.List[float]):
        """
            Função que realiza o processo de Feedfoward na Rede Neural

            Setp-by-Step of Feedfoward:
            ---

            1. Realiza a operação (W_ih•Input)+B_h, onde:
                    * W_ih   -> Matriz de pesos entre entrada e ocultas.
                    * Input  -> Vetor de entrada.
                    * B_h    -> vetor de Bias da camada oculta.
                    * •      -> produto matricial.

            2. Aplica a função S(x) na expressão (W_ih•Input)+B_h, onde:
                    * S(x)   -> Sigmoid function.
                    * x      -> (W_ih•Input)+B_h -> Produto matricial entre Matriz W_ih e Input somado ao B_h.

            3. Realiza a operação (W_ho•Hidden)+B_h, onde:
                    * W_oh   -> Matriz de pesos entre ocultas e saida.
                    * Hidden -> Matrix com os valores das ocultas, aplicado a função S(x) em cada elemento.
                    * B_o    -> vetor de Bias da saida.
                    * •      -> produto matricial.

            4. Aplica a função S(x) na expressão (W_ho•Hidden)+B_o, onde:
                    * S(x)   -> Sigmoid function.
                    * x      -> (W_ho•Hidden)+B_o -> Produto matricial entre Matriz W_oh e Ocultas somado ao B_o.

            ---
            Parametros:
                * input_array: List of Float - Lista que contem o vetor de entrada

            ---
            Retorno:
                * output: 

        """

        # Transforma o vetor de entrada em um np.ndarray
        input_array = mat.matrix.from_array(input_array)

        # Step 1
        hidden = mat.matrix.mat_mul(self.weights_ih.data, input_array)
        hidden = mat.matrix.add_scale(hidden, self.bias_h.data)

        # Step 2
        hidden = mat.matrix.map_matrix(hidden, neural_network.sigmoid)

        # Step 3
        output = mat.matrix.mat_mul(self.weights_ho.data, hidden)
        output = mat.matrix.add_scale(output, self.bias_o.data)

        # Step 4
        output = mat.matrix.map_matrix(output, neural_network.sigmoid)

        del hidden

        return mat.matrix.to_array(output)
