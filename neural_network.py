import numpy as np
from random import random
from warnings import catch_warnings

import math
import typing

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivate(y):
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

def from_array(arr: typing.List[float]) -> np.ndarray:
    """
        Função que converte array em numpy.ndarray.

        ---
        Parametros:
        ---
            * arr - Array[float] - Lista a ser convertida
        ---
        Retornos:
        ---
            * return: numpy.ndarry - Array convertido.
    """
    converted_arr = np.array(arr)
    try:
        converted_arr = np.reshape(arr, (converted_arr.shape[0], converted_arr.shape[1]))
    except:
        converted_arr = np.reshape(arr, (len(arr), 1))

    return converted_arr

def to_array(arr: np.ndarray) -> typing.List[float]:
    """
        Função que converte um Matrix object em uma List[float].
        Normalmente usado para converter a saída da RNA em um array
        ---
        Parametros:
        ---
            * self: Matrix object - Matrix a ser convertida.
        ---
        Retorno:
        ---
            * Array: List[float] - Array convertido em uma lista de floats.
    """
    converted_arr = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            converted_arr.append(arr[i][j])
    return converted_arr


class neural_network:
    def __init__(self, input_nodes, hidden_nodes: int = 0, output_nodes: int = 0):
        """
            Constructor Method
            ---

            Construtor da classe neural_network

            ---
            Parametros:
            ---
                * input_nodes : int            - Numero de neurônios de entrada.
                * hidden_nodes: int, Default=0 - Numero de neurônios da camada oculta.
                * output_nodes: int, Default=0 - Numero de neurônios da camada de saída.
            ---
            Atributos:
            ---
                * input_nodes  : int                - Numero de neurônios da camada de entrada.
                * hidden_nodes : int                - Numero de neurônios da camada oculta.
                * output_nodes : int                - Numero de neurônios da camada de saída.
                * weights_ih   : numpy.ndarray      - Matriz de pesos da camada de entrada e oculta.
                * weights_ho   : numpy.ndarray      - Matriz de pesos da camada oculta e da saída.
                * weights_ho_t : numpy.ndarray      - Matriz de pesos da camada oculta e da saida transposta.
                * bias_h       : numpy.ndarray      - Matriz com os Bias da camada oculta.
                * bias_o       : numpy.ndarray      - Matriz com os Bias da camada de saída.
                * learning_rate: float, Default=0.1 - Indice de aprendizagem da RNA.
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = np.random.rand(input_nodes, hidden_nodes)
        self.weights_ho = np.random.rand(hidden_nodes, output_nodes)
        
        self.bias_h = np.random.rand(hidden_nodes, 1)
        self.bias_o = np.random.rand(output_nodes, 1)
        
        self.learning_rate = 0.1

    def predict(self, input_array: np.ndarray):
        """
            Função que realiza o processo de Feedfoward na Rede Neural
            
            ---
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
            ---
                * self       : neural network object - rede neural que chamou o método.
                * input_array: numpy.ndarray         - Lista que contem o vetor de entrada
            ---
            Retorno:
            ---
                * return: numpy.ndarray - Array com as predições da rede neural. 
        """
        self.hidden = sigmoid(np.dot(input_array, self.weights_ih) + self.bias_h.T)
        outputs = sigmoid(np.dot(self.hidden, self.weights_ho) + self.bias_o)

        return outputs

    def train(self, inputs_array: np.ndarray, targets_array: np.ndarray, max_iter=3500):
        """
            Função que realiza o treino da rede neural.
            
            ---
            Step-by-Step of Train:
            ---
            1.  Obtém a saída com o processo de Feedfoward.\n
            2.  Calcula-se o erro da matriz de pesos associada a camada oculta N e a camada de saída. 
                Aplicando a seguinte operação:`(2 * (y_real - y_obtido) * d_S(y_obtido))`, onde:\n
                    * y_real    -> valor real(valor com a label classifica).\n
                    * y_obtido  -> valor obtido pela classificação da rede neural.\n
                    * d_S(x)    -> função da derivada de sigmoid.\n
                    * x         -> valores que serão aplicado a função da derivada de `sigmoid` (y_obtido).\n
            3.  Calcula-se o gradiente que será usado para atualizar o valor da Bias associada a saída. 
                Aplicando a seguinte operação:`d_S(y_obtido) * errors_ho * learning_rate`, onde:
                    * d_S(x)        -> função da derivada de sigmoid.
                    * x             -> valores que serão aplicados a função da derivada de sigmoid (y_obtido).
                    * y_obtido      -> valor obtido pela classificação da rede neural.
                    * errors_ho     -> valor obtido no `passo 2`.
                    * learning_rate -> taxa de aprendizagem `(Default=0.1)`.
            4.  Repete os `Passos 2 e 3` até o começo da rede neural.

            ---
            Parametros:
            ---

                * self          : neural network object - rede neural que chamou o método.
                * inputs_array  : numpy.ndarray - Array que contem os dados de treino
                * targets_array : numpy.ndarray - Array que contem a classificação/valor do input_array.
            ---
            Retorno:
            ---
                * None - Não possui retorno pois, trabalha com referencia.
              
        """

        inputs_array = from_array(inputs_array)
        targets_array = from_array(targets_array)

        for i in range(max_iter):
            # Feedfoward - Para obter a saida e calcular os valores da camada oculta.
            outputs = self.predict(inputs_array)

            # Calcula o erro da camada oculta e da camada de saída, pela formula:
            # erro = (2 * (y_real - y_estimado) * sigmoid_derivate(y_estimado))
            errors_ho = (2 * (targets_array - outputs) * sigmoid_derivate(outputs))

            # Calculando o gradient
            # gradient = derivada de sigmoid nos outputs * erro * taxa de aprendizado
            gradients = sigmoid_derivate(outputs)
            gradients = gradients * errors_ho
            gradients = gradients * self.learning_rate

            # Como o treinamento é realizado com um conjunto > 1, então é pegado a média dos erros.
            mean_gradients = np.mean(gradients)

            # Calculado a nova matriz de pessos associada entre a oculta e saída
            weights_ho = np.dot(self.hidden.T, errors_ho)

            # Ajustado o bias pelos deltas (que no caso é apenas o gradiente)
            self.bias_o = self.bias_o + mean_gradients

            # Calcula o erro da camada de entrada e da camada oculta.
            errors_ih = (2 * (targets_array - outputs) * sigmoid_derivate(outputs))  
            errors_ih = np.dot(errors_ih, self.weights_ho.T) * sigmoid_derivate(self.hidden)

            # gradient = derivada de sigmoid nos outputs * erro;
            # Calculate gradient
            gradients = sigmoid_derivate(self.hidden)
            gradients = gradients * errors_ih
            gradients = gradients * self.learning_rate
            # Como o treinamento é realizado com um conjunto > 1, então é pegado a média dos erros.
            mean_gradients = np.mean(gradients)

            # Calculado a nova matriz de pessos associada entre a camada de entrada e camada oculta
            weights_ih = np.dot(inputs_array.T,  errors_ih)

            # Ajustado o bias pelos deltas (que no caso é apenas o gradiente)
            self.bias_h = self.bias_h + mean_gradients

            # Atualiza as matrizes de pesos com as derivadas (declive) da função de perda
            self.weights_ih += weights_ih
            self.weights_ho += weights_ho

    def mutate(self, rate: float):
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
