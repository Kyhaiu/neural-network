import typing
import numpy as np


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

    def mat_randomize(self) -> None:
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
                self.data[i][j] = np.random.uniform(-1, 1)
                j += 1
            i += 1
        del i, j

    @staticmethod
    def from_array(arr: typing.List[float]) -> np.ndarray:
        """
            Função que converte um array em um numpy.ndarray.
            Normalmente usada para receber inputs do usuário através de um array,
            e converte em um numpy.ndarray de 1-d.

            ---
            Parametros:
                * arr: List of float - Array que contem os dados.

            ---
            Retorno:
                * Array: 1-d numpy.ndarray - Array convertido em um ndarray de 1-d.
        """
        array_converted = np.ndarray((1, len(arr)))

        i = 0
        while i < len(arr):
            array_converted[0][i] = arr[i]
            i += 1

        del i
        return array_converted.reshape((len(arr), 1))

    @staticmethod
    def to_array(arr: np.ndarray) -> typing.List[float]:
        """
            Função que converte um numpy.ndarray em uma List[float].
            Normalmente usado para converter a saída da RNA em um array

            ---
            Parametros:
                * arr: numpy.ndarray - Array que contem os dados.

            ---
            Retorno:
                * Array: List[float] - Array convertido em uma lista de floats.
        """
        it = np.nditer(arr)
        array_converted = []
        for it in arr:
            array_converted.append(it[0])

        del it
        return array_converted

    @staticmethod
    def mat_add_elementwise(matrix: np.ndarray, scalar) -> np.ndarray:
        """
            Função que realiza a operação Elementwise de adição na matriz.

            ---
            Parametros:
                * matrix: numpy.ndarray - Matriz(mxn).
                * scalar: Integer or numpy.ndarray(mxn) - Escalar que sera adicionado.

            ---
            Retorno:
                * Matriz: numpy.ndarray - Matriz.
        """
        return np.add(matrix, scalar)

    @staticmethod
    def mat_sub_elementwise(matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray:
        """
            Função que realiza a operação Elementwise de subtração na matriz.

            ---
            Parametros:
                * matrixA: numpy.ndarray - Matriz A
                * matrixB: numpy.ndarray - Matriz B

            ---
            Retorno:
                * Matriz: numpy.ndarray - Matriz C.
        """
        return np.subtract(matrixA, matrixB)

    @staticmethod
    def mat_mul_elementwise(matrix: np.ndarray, scalar) -> np.ndarray:
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

    @staticmethod
    def mat_mul_dotproduct(matrixA: np.ndarray, matrixB: np.ndarray) -> np.ndarray:
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
        return np.dot(matrixA, matrixB)

    @staticmethod
    def transpose(matrix: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def mat_map(matrix: np.ndarray, func) -> np.ndarray:
        i, j = 0, 0
        rows, cols = matrix.shape[0], matrix.shape[1]
        while i < rows:
            j = 0
            while j < cols:
                aux = matrix[i][j]
                matrix[i][j] = func(aux)
                j += 1
            i += 1
        del i, j, rows, cols, aux
        return matrix
