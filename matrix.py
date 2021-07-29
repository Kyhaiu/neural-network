import typing

import numpy as np
from random import uniform
from numpy.core.fromnumeric import shape
from numpy.matrixlib import mat


class matrix:

    def __init__(self,  rows: int, cols: int) -> None:
        """
            Construtor da classe Matrix.
            Cria uma matriz(mxn) preenchida com valores 0.
            ---
            Parametros:
                * rows: Integer - Numero de linhas da matriz.
                * cols: Integer - Numero de colunas da matriz.

            ---
            Retorno:
                Matriz: numpy.ndarray - Matriz gerada.
        """
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((rows, cols))

    def __setattr__(self, name: str, value: typing.Any) -> None:
        """
            Definição de atributos da classe Matrix

            ---
            Atributos:
                * rows: Integer - Numero de linhas da matriz(m).
                * cols: Integer - Numero de colunas da matriz(n).
                * data: numpy.ndarray - Valores da matriz(mXn).

            ---
        """
        if name == 'rows':
            super(matrix, self).__setattr__(name, value)
        if name == 'cols':
            super(matrix, self).__setattr__(name, value)
        if name == 'data':
            super(matrix, self).__setattr__(name, value)

    @staticmethod
    def from_array(input_array: typing.List[float]) -> np.ndarray:
        """
            Função que converte um array em um numpy.ndarray.
            Normalmente usada para receber inputs do usuário através de um array,
            e converte em um numpy.ndarray de 1-d.

            ---
            Parametros:
                * input_array: List of float - Array que contem os dados.

            ---
            Retorno:
                * Array: 1-d numpy.ndarray - Array convertido em um ndarray de 1-d.
        """
        return np.array(input_array)

    def mul_scale(self, matrix: np.ndarray, scalar) -> np.ndarray:
        """
            Função que realiza a operação Elementwise de multiplicação na matriz

            ---
            Parametros:
                * matrix: numpy.ndarray - Matriz(nxm).
                * scalar: Integer or numpy.ndarray - Escalar que a matriz será multiplicada.

            ---
            Retorno:
                * Matriz: numpy.ndarray - Matriz(mxn) escalonada.
        """
        matrix.data = np.multiply(matrix.data, scalar)

    def add_scale(self, matrix: np.ndarray, scalar) -> np.ndarray:
        """
            Função que realiza a operação Elementwise de adição na matriz.

            ---
            Parametros:
                * matrix: numpy.ndarray - Matriz(mxn).
                * scalar: Integer or numpy.ndarray(mxn) - Escalar que sera adicionado.

            ---
            Retorno:
                * Matriz: numpy.ndarray - Matriz(mxn) escalonada.
        """
        matrix.data = np.add(matrix.data, scalar)

    def mat_mul(self, matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray:
        """
            Função que realiza o produto matricial entre duas matrizes

            ---
            Parametros:
                * matrixA: numpy.ndarray - Matriz(mxk)
                * matrixB: numpy.ndarray - Matrix(kxn)

            ---
            Retorno:
                * Matrix: numpy.ndarray - Matriz(mxn)
        """
        return np.matmul(matrixA.data, matrixB.data)

    def transpose(self, matrix: np.ndarray) -> np.ndarray:
        """
            Função que realiza a transposição da matriz.

            ---
            Parametros:
                * matrix: numpy.ndarray - Matriz(mxn).

            ---
            Retorno:
                * matrix: numpy.ndarray - Matriz(nxm).
        """
        return matrix.data.T

    def randomize(self) -> None:
        """
            Função que preenche a matriz gerada com valores aleatórios entre -1 e 1

            ---
            Parametros:
                * None - Não possui parametros.

            ---
            Retorno:
                * None - Não possui retorno pois trabalha com referencia.
        """
        i = 0
        j = 0
        while i < self.rows:
            j = 0
            while j < self.cols:
                self.data[i][j] = uniform(-1, 1)
                j += 1
            i += 1
        del i, j
