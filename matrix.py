import typing

import numpy as np
from random import uniform
from numpy.matrixlib import mat


class matrix:

    def __init__(self,  rows: int, cols: int) -> None:
        """
            Construtor da classe matrix.
            Cria uma matriz de tamanho mXn preenchida com valores 1.
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
        self.data = np.ones((rows, cols))

    def __setattr__(self, name: str, value: typing.Any) -> None:
        """
            Definição de atributos da classe matrix

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

    def randomize(self, matrix: np.ndarray) -> np.ndarray:
        """
            Função que preenche a matriz gerada com valores aleatórios entre -1 e 1

            ---
            Parametros:
                * Matrix: numpy.ndarray - Matriz(mxn) que será randomizada.

            ---
            Retorno:
                * Matrix: numpy.ndarray - Matriz(mxn) randomizada.
        """
        i = 0
        j = 0
        while i < matrix.shape[0]:
            j = 0
            while j < matrix.shape[1]:
                matrix[i][j] = uniform(-1, 1)
                j += 1
            i += 1
        del i, j
        return matrix

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

        return np.multiply(matrix, scalar)

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
        return np.add(matrix, scalar)

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
        return np.matmul(matrixA, matrixB)

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
        return matrix.T
