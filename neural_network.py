
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
