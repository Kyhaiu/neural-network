from random import random
import typing


class Matrix:
    def __init__(self, rows: int, cols: int):
        """
            Construtor da classe Matrix.
            Cria uma matriz(mxn) preenchida com valores 0.
            ---
            Parametros:
                * rows: Integer - Numero de linhas da matriz.
                * cols: Integer - Numero de colunas da matriz.
            ---
            Retorno:
                Matriz: List - Matriz gerada.
        """
        self.rows = rows
        self.cols = cols
        self.data = []

        for i in range(self.rows):
            self.data.append([])
            for _ in range(self.cols):
                self.data[i].append(0)

    def __str__(self):
        """
            String customizada da classe Matrix.
            Usado para printar na tela objeto.data
        """
        return str(self.data)

    @staticmethod
    def fromArray(arr: typing.List[float]):
        """
            Satic Method
            ---

            Função que converte um array em uma matriz.
            Normalmente usada para receber inputs do usuário através de um array,
            e converter em um matriz de 1-d.
            ---
            Parametros:
                * arr: List of float - Array que contem os dados.
            ---
            Retorno:
                * m: Matrix object - Matrix de 1-d convertida.
        """
        m = Matrix(len(arr), 1)
        for i in range(len(arr)):
            m.data[i][0] = arr[i]
        return m

    def toArray(self):
        """
            Função que converte um Matrix object em uma List[float].
            Normalmente usado para converter a saída da RNA em um array
            ---
            Parametros:
                * self: Matrix object - Matrix a ser convertida.
            ---
            Retorno:
                * Array: List[float] - Array convertido em uma lista de floats.
        """
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

    def randomize(self):
        """
            Função que preenche a matriz gerada com valores aleatórios entre -1 e 1
            ---
            Parametros:
                * self: Matrix object - Matriz que será preenchida com valores aleatórios 
            ---
            Retorno:
                * None - Não possui retorno pois trabalha com referencia.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                r = random()
                r = r if random() < 0.5 else r - 1
                self.data[i][j] = r

    @staticmethod
    def subtract(a, b):
        """
            Static Method
            ---

            Função que realiza a operação Elementwise de subtração na matriz.

            ---
            Parametros:
                * a: Matrix object - Matriz A
                * b: Matrix object - Matriz B
            ---
            Retorno:
                * return: Matrix object - Matriz C resultado da subtração Elementwise.
        """
        result = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result

    def add(self, n):
        """
            Função que realiza a operação Elementwise de adição na matriz.

                * Se n é um instancia de Matrix object, então é realizado a operação soma das matrizes
                * Se n é um int/float então é realizado a operação de adição elementwise do escalar

            ---
            Parametros:
                * self: Matrix object - Matriz A.
                * n   : Matrix object or int/float - Matriz B/Escalar que será somado a Matriz A
            ---
            Retorno:
                * None - Não possui retorno pois trabalha com referencia.
        """
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    @staticmethod
    def transpose(matrix):
        """
            Static Method
            ---

            Função que realiza a transposição da matriz.

            ---
            Parametros:
                * Matrix: Matrix object - Matriz(mxn).
            ---
            Retorno:
                * return: Matrix object - Matriz(nxm).
        """
        result = Matrix(matrix.cols, matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[j][i] = matrix.data[i][j]
        return result

    # Matrix Multiplication
    @staticmethod
    def multiply1(a, b):
        """
            Static Method
            ---

            Função que realiza a multiplicação entre duas matrizes.

            ---
            Parametros:
                * a: matriz object - Matriz(mxn) A.
                * b: matriz object - Matriz(nxm) B.
            ---
            Retorno:
                * return: Matrix object - Matriz(mxn) C resultante da multiplicação.
        """
        if a.cols != b.rows:
            print('Invalid Matrices!')
            return

        result = Matrix(a.rows, b.cols)

        for i in range(result.rows):
            for j in range(result.cols):
                s = 0.0
                for k in range(a.cols):
                    s += a.data[i][k] * b.data[k][j]
                result.data[i][j] = s
        return result

    def multiply2(self, n):
        """
            Função que realiza o produto escalar ou Hadamard product

            ---
            Parametros:
                * self: Matrix object - Matrix A.
                * n: Matrix object or int/float - Matriz B/Escalar

            ---
            Retorno:
                * Se n é um Matrix object, então retorna o Hadamard Product na Matrix object que chamou o método.
                * Se n é um escalar, então retorna o Produto escalar na Matrix object que chamou o método.
        """
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                print('Invalid Matrices!')
                return
            # Hadamard Product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            # Scalar Product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    def map1(self, func):
        """
            Função que aplica uma função em todos os elementos da Matrix object.

            ---
            Parametros:
                * self: Matrix object - Matriz que será aplicada a função
                * func: function reference - Referencia da função que será aplicada.

            ---
            Retorno:
                * None - Não possui retorno pois trabalha com referencia.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j])

    @staticmethod
    def map2(m, func):
        """
            Static Method
            ---

            Função que aplica uma função em todos os elementos da Matrix object.

            ---
            Parametros:
                * m: Matrix object - Matriz que será aplicada a função
                * func: function reference - Referencia da função que será aplicada.

            ---
            Retorno:
                * return: Matrix object - Matrix M com a função aplicada.
        """
        result = Matrix(m.rows, m.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = func(m.data[i][j])
        return result

    def copy(self):
        """
            Função responsável por copiar os dados de uma Matrix object para outro Matrix object.

            ---
            Parametro:
                * self: Matrix object - Matrix que chamou o método

            ---
            Retorno:
                * return: Matrix object - Copia da Matrix object.
        """
        m = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                m.data[i][j] = self.data[i][j]
        return m
