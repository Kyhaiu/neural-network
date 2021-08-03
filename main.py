import numpy as np
import neural_network as nn
from sklearn.neural_network import MLPClassifier
import snake

"""
    Explicar o projeto de como treinar o PONG e ver se ta dentro das especificações

    Para ajustar o bias estou utilizado o erro médio, já que só temos 1 output. Isso pode?
    ou tenho que treinar elemento a elemento e ir ajustando o bias?

    como calcular o erro de cada camdas que não sejam input ou output
"""


class Main():

    def __init__(self) -> None:
        pass

    def test(self):

        snake.init()
        """rna = nn.neural_network(2, (2, 4), 1)
        x_train, y_train = [], []
        x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_train = [0, 1, 1, 0]
        
        rna.train(x_train, y_train, 3500)

        print("my rna guess for [0,0]:", nn.to_array(rna.predict([0,0])))
        print("my rna guess for [0,1]:", nn.to_array(rna.predict([0,1])))
        print("my rna guess for [1,0]:", nn.to_array(rna.predict([1,0])))
        print("my rna guess for [1,1]:", nn.to_array(rna.predict([1,1])))"""

        #mlp = MLPClassifier(hidden_layer_sizes=(2, 2), activation='logistic', learning_rate='constant', max_iter=50000)
        #mlp.fit(x_train, y_train)

        #print("mlp guess:", mlp.predict([[0,0], [0,1], [1,0], [1,1]]))


main = Main()
main.test()
