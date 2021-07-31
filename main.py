import neural_network as nn
import random
from sklearn.neural_network  import MLPClassifier


class Main():

    def __init__(self) -> None:
        pass

    def test(self):
        rna = nn.neural_network(2, 2, 1)
        training_data = [([0, 1], [1]), ([1, 0], [1]),
                         ([0, 0], [0]), ([1, 1], [0])]
        i = 0

        x_train, y_train = [], []

        while i < 100000:
            data_learning = random.sample(training_data, 1)
            rna.train(data_learning[0][0], data_learning[0][1])
            #x_train.append(data_learning[0][0])
            #y_train.append(data_learning[0][1][0])
            i += 1

        print("my rna guess for [0,0]:", rna.predict([0,0]))
        print("my rna guess for [0,1]:", rna.predict([0,1]))
        print("my rna guess for [1,0]:", rna.predict([1,0]))
        print("my rna guess for [1,1]:", rna.predict([1,1]))

        #mlp = MLPClassifier(hidden_layer_sizes=(2, 1), activation='logistic', learning_rate='constant', max_iter=1000)
        #mlp.fit(x_train, y_train)

        #print(mlp.get_params)
        #print("mlp guess:", mlp.predict([[0,0], [0,1], [1,0], [1,1]]))


main = Main()
main.test()