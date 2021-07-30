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

        self.weights_ih.mat_randomize()
        self.weights_ho.mat_randomize()

        self.bias_h = mat.matrix(self.hidden_nodes, 1)
        self.bias_o = mat.matrix(self.output_nodes, 1)

        self.bias_h.mat_randomize()
        self.bias_o.mat_randomize()

        self.learning_rate = 0.1

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
        if name == 'learning_rate':
            super(neural_network, self).__setattr__(name, value)

    def sigmoid(x) -> float:
        try:
            return (1/(1+math.exp(-x)))
        except:
            return (1/(1+math.exp(-709)))

    def derivate_sigmoid(y) -> float:
        return (y - (1 - y))

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
        hidden = mat.matrix.mat_mul_dotproduct(
            self.weights_ih.data, input_array)
        hidden = mat.matrix.mat_add_elementwise(
            hidden, self.bias_h.data)

        # Step 2
        hidden = mat.matrix.mat_map(hidden, neural_network.sigmoid)

        # Step 3
        output = mat.matrix.mat_mul_dotproduct(self.weights_ho.data, hidden)
        output = mat.matrix.mat_add_elementwise(output, self.bias_o.data)

        # Step 4
        output = mat.matrix.mat_map(output, neural_network.sigmoid)

        del hidden

        return mat.matrix.to_array(output)

    def train(self, input_array: typing.List[float], target_array: typing.List[float]):
        """
            Função que realiza o treino da rede neural.

            Step-by-Step of Train:
            ---

            1. Obtém a a saída da rede neural com base nos inputs
                    * Realiza a operação feedfoward

            2. Calcula o erro absoluto da saída obtida, aplicando a operação E_mat = (T_mat - O_mat), onde:
                    * E_mat -> Matriz que irá receber os erros absolutos.
                    * T_mat -> Matriz que contem o gabarito das respostas esperadas.
                    * O_mat -> Matriz que contem a saída da rede neural.

            3. Calcula o Erro das camadas ocultas, aplicando a operação (W_ho_T•E_mat), onde:
                    * W_ho_T -> Matriz dos pesos entre camada oculta e saída transposta.
                    * E_mat  -> Matriz que contem os erros absolutos da rede neural.
                    * •      -> Produto matricial.


        """

        # Step 1.
        inputs = mat.matrix.from_array(input_array)

        # Step 1.1 - Feedfoward
        hiddens = mat.matrix.mat_mul_dotproduct(self.weights_ih.data, inputs)
        hiddens = mat.matrix.mat_add_elementwise(hiddens, self.bias_h.data)

        # Step 1.2 - Feedfoward
        hiddens = mat.matrix.mat_map(hiddens, neural_network.sigmoid)

        # Step 1.3 - Feedfoward
        outputs = mat.matrix.mat_mul_dotproduct(self.weights_ho.data, hiddens)
        outputs = mat.matrix.mat_add_elementwise(outputs, self.bias_o.data)

        # Step 1.4 - Feedfoward
        outputs = mat.matrix.mat_map(outputs, neural_network.sigmoid)

        ########################

        # Step 2.
        targets = mat.matrix.from_array(target_array)

        output_errors = mat.matrix.mat_sub_elementwise(targets, outputs)

        # (outputs - (1 - outputs)) -> Derivada da função sigmoid
        gradients = mat.matrix.mat_map(
            outputs, neural_network.derivate_sigmoid)
        gradients = mat.matrix.mat_mul_elementwise(gradients, output_errors)
        gradients = mat.matrix.mat_mul_elementwise(
            gradients, self.learning_rate)

        hidden_T = mat.matrix.transpose(hiddens)
        weights_ho_deltas = mat.matrix.mat_mul_dotproduct(gradients, hidden_T)

        # Ajustando os pesos da camada oculta e saída pesos pelos deltas
        self.weights_ho.data = mat.matrix.mat_add_elementwise(
            self.weights_ho.data, weights_ho_deltas)
        # Ajustando o bias do output pelos deltas (que no caso é o gradient)
        self.bias_o.data = mat.matrix.mat_add_elementwise(
            self.bias_o.data, gradients)

        # Step 3.
        weights_ho_transpose = mat.matrix.transpose(self.weights_ho.data)
        hidden_errors = mat.matrix.mat_mul_dotproduct(
            weights_ho_transpose, output_errors)

        # Calcular o gradiente da camada oculta
        hidden_gradients = mat.matrix.mat_map(
            hiddens, neural_network.derivate_sigmoid)
        hidden_gradients = mat.matrix.mat_mul_elementwise(
            hidden_errors, hidden_errors)
        hidden_gradients = mat.matrix.mat_mul_elementwise(
            hidden_gradients, self.learning_rate)

        # Calculate input->hidden deltas
        inputs_T = mat.matrix.transpose(inputs)
        weights_ih_deltas = mat.matrix.mat_mul_dotproduct(
            hidden_gradients, inputs_T)

        # Ajustando os pesos da camada de entrada e oculta pesos pelos deltas
        self.weights_ih.data = mat.matrix.mat_add_elementwise(
            self.weights_ih.data, weights_ih_deltas)
        # Ajustando o bias do output pelos deltas (que no caso é o hidden_gradients)

        self.bias_h.data = mat.matrix.mat_add_elementwise(
            self.bias_h.data, hidden_gradients)
