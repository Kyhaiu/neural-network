import numpy as np
from random import uniform


class NeuralNetwork:

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int) -> None:
        """
            Construtor da classe NeuralNetwork
            ---
            Parametros:
                * input_nodes: Integer - Numero de neurônios de entrada
                * hidden_nodes: Integer - Numero de neurônios da camada oculta
                * output_nodes: Integer - Numero de neurônios da camada de saída
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

    def matrix(self, rows: int, cols: int) -> np.ndarray:
        """
            Função que cria uma matriz de tamanho mXn preenchida com valores 1.
            ---
            Parametros:
                * rows: Integer - Numero de linhas da matriz.
                * cols: Integer - Numero de colunas da matriz.
            Retorno:
                Matriz: numpy.ndarray - Matriz gerada.
        """
        self.rows = rows
        self.cols = cols
        self.matrix = np.ones((self.rows, self.cols))
        return self.matrix

    def randomize(self):
        """
            Função que preenche a matriz gerada com valores aleatórios entre -1 e 1
            ---
            Parametros:
                * None - Não possui nenhum parametro, pois trabalha por referencia.

            Retorno:
                * None - Não retorna nada, pois trabalha por referencia.
        """
        i = 0
        j = 0
        while i < self.matrix.shape[0]:
            j = 0
            while j < self.matrix.shape[1]:
                self.matrix[i][j] = uniform(-1, 1)
                j += 1
            i += 1
        del i, j

    def mul_scale(self, scalar) -> np.ndarray:
        """
            Função que realiza a operação Elementwise de multiplicação na matriz
            ---
            Parametros:
                * scalar: Integer or numpy.ndarray - Escalar que a matriz será multiplicada.

            Retorno:
                * Matriz: numpy.ndarray - Matriz escalonada.
        """

        return np.multiply(self.matrix, scalar)

    def add_scale(self, scalar) -> np.ndarray:
        """
            Função que realiza a operação Elementwise de adição na matriz.
            ---
            Parametros:
                * scalar: Integer or numpy.ndarray - Escalar que sera adicionado.

            Retorno:
                * Matriz: numpy.ndarray - Matriz escalonada.
        """
        return np.add(self.matrix, scalar)
