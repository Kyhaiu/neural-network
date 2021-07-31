import math
from matrix import Matrix
from random import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


# Random Gaussian
y2 = None
previous = False


def randomGaussian(mean=0, sd=1):
    y1, x1, x2, w = None, None, None, None
    global previous, y2
    if previous:
        y1 = y2
        previous = False
    else:
        while True:
            x1 = random() * 2 - 1
            x2 = random() * 2 - 1
            w = x1 * x1 + x2 * x2
            if w < 1:
                break
        w = math.sqrt((-2 * math.log(w)) / w)
        y1 = x1 * w
        y2 = x2 * w
        previous = True

    return y1 * sd + mean


class neural_network:
    def __init__(self, input_nodes, hidden_nodes: int = 0, output_nodes: int = 0):
        """
            Constructor Method
            ---

            Construtor da classe NeuralNetwork

            ---
            Parametros:
                * input_nodes : int            - Numero de neurônios de entrada.
                * hidden_nodes: int, Default=0 - Numero de neurônios da camada oculta.
                * output_nodes: int, Default=0 - Numero de neurônios da camada de saída.
            ---
            Atributos:
                * input_nodes  : int                - Numero de neurônios da camada de entrada.
                * hidden_nodes : int                - Numero de neurônios da camada oculta.
                * output_nodes : int                - Numero de neurônios da camada de saída.
                * weights_ih   : Matrix object      - Matriz de pesos da camada de entrada e oculta.
                * weights_ho   : Matrix object      - Matriz de pesos da camada oculta e da saída.
                * weights_ho_t : Matrix object      - Matriz de pesos da camada oculta e da saida transposta.
                * bias_h       : Matrix object      - Matriz com os Bias da camada oculta.
                * bias_o       : Matrix object      - Matriz com os Bias da camada de saída.
                * learning_rate: float, Default=0.1 - Indice de aprendizagem da RNA.
        """
        if not hidden_nodes:
            a = input_nodes
            self.input_nodes = a.input_nodes
            self.hidden_nodes = a.hidden_nodes
            self.output_nodes = a.output_nodes
            self.weights_ih = a.weights_ih.copy()
            self.weights_ho = a.weights_ho.copy()
            self.weights_ho_t = a.weights_ho_t.copy()
            self.bias_h = a.bias_h.copy()
            self.bias_o = a.bias_o.copy()
        else:
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
            self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
            self.weights_ho_t = Matrix.transpose(self.weights_ho)
            self.bias_h = Matrix(self.hidden_nodes, 1)
            self.bias_o = Matrix(self.output_nodes, 1)

        self.weights_ih.randomize()
        self.weights_ho.randomize()
        self.bias_h.randomize()
        self.bias_o.randomize()
        self.learning_rate = 0.1

    def predict(self, input_array):
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
                * self       : neural network object - rede neural que chamou o método.
                * input_array: Array[float] - Lista que contem o vetor de entrada
            ---
            Retorno:
                * return: Array[float] - Array com as predições da rede neural. 
        """
        # Computing Hidden Outputs
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.multiply1(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map1(sigmoid)  # Activation Function

        # Computing Output Layer's Output!
        outputs = Matrix.multiply1(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map1(sigmoid)

        return outputs.toArray()

    def train(self, input_array, target_array):
        """
            Função que realiza o treino da rede neural.
            Step-by-Step of Train:
            ---
            Elaborar o setp-by-setp
        """
        # Computing Hidden Outputs
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.multiply1(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map1(sigmoid)  # Activation function

        # Computing Output Layer's Output!
        outputs = Matrix.multiply1(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map1(sigmoid)  # Neural Net's Guess

        # Converting target array to Matrix object
        targets = Matrix.fromArray(target_array)

        # Calculate Error
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate Hidden Errors
        hidden_errors = Matrix.multiply1(self.weights_ho_t, output_errors)

        # Calculate gradients
        gradients = Matrix.map2(outputs, dsigmoid)
        gradients.multiply2(output_errors)
        gradients.multiply2(self.learning_rate)

        # Calculate Deltas
        hidden_t = Matrix.transpose(hidden)
        weight_ho_deltas = Matrix.multiply1(gradients, hidden_t)

        # Adjust Hidden -> Output weights and output layer's biases
        self.weights_ho.add(weight_ho_deltas)
        self.bias_o.add(gradients)

        # Calculate the hidden gradients
        hidden_gradients = Matrix.map2(hidden, dsigmoid)
        hidden_gradients.multiply2(hidden_errors)
        hidden_gradients.multiply2(self.learning_rate)

        # Calculate Deltas
        input_t = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.multiply1(hidden_gradients, input_t)

        # Adjust Input -> Hidden weights and hidden biases
        self.weights_ih.add(weight_ih_deltas)
        self.bias_h.add(hidden_gradients)

    def mutate(self, rate):
        """
            Função que realiza mutação nos pesos e nas Bias da rede neural

            ---
            Parametros:
                * self: neural network object - Rede neural que chamou a função.
                * rate: float                 - Taxa de mutação
            
            ---
            Retorno:
                * None - Não possui retorno pois trabalha com referencia.
        """
        def mutate(x): return x + randomGaussian(0, 0.1) if random() < rate else x
        self.weights_ih.map1(mutate)
        self.weights_ho.map1(mutate)
        self.bias_h.map1(mutate)
        self.bias_o.map1(mutate)

    def copy(self):
        """
            Função para criar uma cópia da sí mesmo.

            ---
            Parametros:
                * self: neural network object - Rede neural que chamou a função

            ---
            Retornos:
                * return: neural network object - Cópia da rede neural que chamou a função.
        """
        return neural_network(self)
