import sys
import cv2
import os
import numpy as np
import keras.backend as kb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Input


path = './data'

# Dataset final
xtotal = []
ytotal = []

#Lista de aquivos no diretório /data
filenames = os.listdir(path)

# O Dataset terá no max 200000 elementos.
# O numero max de cada elementos de um conjunto será a média no max.
# A divisão por 2 é que na lista de filenames estão sendo considerados os .rar
slice_train = int(200000/(len(filenames)/2))

# Seed usada para embaralhar os dados antes de dividir
seed = np.random.randint(1, 10e7)

# Mapa de rótulos dos conjuntos
label_map = {}

i = 0
for fname in filenames :
    
    # Caso o conjunto não seja do formato proprio para o np.load, ele só ignora.
    if '.npy' not in fname :
        continue

    x = np.load(path + '/' +fname)
    
    # realiza uma escala nos dados de 0-255 para 0-1
    x = x.astype('float32') / 255
    # Pega o nome do conjunto(nome indicado pelo nome do aquivo)
    label_name = fname.split('.npy')[0]

    # Replica o rotúlo obtido em um vetor com o mesmo numero de elementos que tem em x.
    y = [str(label_name)] * len(x)
    
    # Adiciona a label para o map de labels.
    label_map[i] = label_name
    
    # Embaralha e divide os conjuntos de dados x, y.
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)

    # Atribui os primeiros 'slice_train' nos conjuntos x, y.
    # 'slice_train' é o numero maximo de elemento que cada conjunto pode ter.
    x = x[:slice_train]
    y = y[:slice_train]
    
    # Se for o 1º arquivo a ser embaralhado e dividio, então ele só atribui ao xtotal, ytotal os vetores.
    # Caso contrário, ele concatena no conjunto te treino xtotal, ytotal. 
    if i == 0 :
        xtotal = x
        ytotal = y
    else :
        xtotal = np.concatenate([x,xtotal], axis=0)
        ytotal = np.concatenate([y,ytotal], axis=0)
    
    i += 1

# Realiza a divisão utilizando a proporção de 20% para teste, 70% para treino.
# random_state aplica um novo shuffle e mantem a proporção nos conjuntos resultantes.
# 42 pq é a resposta sobre a vida, o universo e tudo mais...
x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.2, random_state=42)

# Realiza o reshape dos conjuntos de teste. De (N, 748) => (N, 28, 28), onde N é o numero de elementos de treino(80% do conjunto total).
# Todos os desenhos simplificados, foram rendenizados em uma matrix 28x28 em grayscale bitmap no formato .npy
# Código aplicado na geração dos arquivos https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

# Processo de transformar os rótulos obtidos em numeros.
encoder = LabelEncoder()
encoded_y_train = encoder.fit_transform(y_train)
y_train_categorical = to_categorical(encoded_y_train)

encoded_y_test = encoder.fit_transform(y_test)
y_test_categorical = to_categorical(encoded_y_test)


def create_rna(num_classes, input_shape, num_hiddens=3, num_neurons=64):
    """
        Função responsável por criar uma rede neural artificial.
        
        Parametros:
        ---
            - num_classes: Integer - Numero de classes que a rede neural deve classificar.
            - input_shape: Tuple of Integers - Tupla que contem 3 numeros inteiros que correspondem ao formato dos dados que a rede neural irá receber como entrada.
            - num_hiddens: Integer - Numero de camdas ocultas que a rede neural terá (Camdas de entrada e saída não são consideradas neste numero.).
            - num_neurons: Integer - Numero de neurônios que as camadas ocultas terão.
        
        ---
        Retorno:
        ---
            - RNA: `tf.keras.Model` - Rede neural artificial.
    """
    rna = Sequential()
    rna.add(Input(input_shape))
    rna.add(Flatten())

    for i in range(num_hiddens):
        rna.add(Dense(num_neurons, activation='relu'))

    rna.add(Dense(num_classes, activation='softmax'))

    rna.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return rna

# Número de classes
number_class = len(label_map)

# Tamanho das imagens do dataset
width = height = 28

# Profundiade dos dados.
depth = 1

input_shape = (depth, width, height)
# O Keras necessita da informação depth, pois se não o shape dos dados ficaria (None, width, height), causando um erro no backend do keras.
# Não entendo o por que ele precisa da profundidade, mas se não colocar da erro. :\/
train_features = x_train.reshape(x_train.shape[0], depth, width, height).astype('float32')
test_features = x_test.reshape(x_test.shape[0], depth, width, height).astype('float32')

train_labels = y_train_categorical
test_labels = y_test_categorical

# Menu de navegação do programa.
if os.listdir('./Model'):
    print('RNA encontrada.')
    keypress = input('Gostaria de treinar uma nova RNA? (Y/n) ')
    if keypress == 'n' or keypress == 'N':
        print("Carregando RNA...")
        rna = load_model('./Model/rede_neural.tf')
    elif keypress == "" or keypress == "Y" or keypress == "y":
        print("Treinando uma nova RNA...")

        rna = create_rna(number_class, input_shape, 3, 64)
        rna.fit(train_features, train_labels, batch_size=100, epochs=3)
        rna.save('./Model/rede_neural.tf')
    else:
        print("Resposta inválida.")
        print("Respostas válidas: [Y-y-enter/N-n]")
        print("Encerrando Programa...")
        print("Liberando Memoria...")
        sys.exit()
else:
    print("Não existe RNA, Criando uma nova...")

    rna = create_rna(number_class, input_shape, 3, 64)
    rna.fit(train_features, train_labels, batch_size=100, epochs=3)
    rna.save('./Model/rede_neural.tf')

eval_results = rna.evaluate(test_features, test_labels)
print('Loss: ', eval_results[0], 'Accuracy: ', eval_results[1])

# Deleção de variaveis não mais utilizadas, para liberar memoria
del x_train, y_train, x_test, y_test, encoded_y_train, encoded_y_test, xtotal, ytotal 

i = 0
y_true = []
y_pred = []
pred = rna.predict(test_features)

# Processo de obter a classe predita, com base no conjunto de teste
while i < len(pred):
    y_pred.append(np.argmax(pred[i], axis=0))
    y_true.append(np.argmax(test_labels[i], axis=0))
    i += 1

del pred, test_features, test_labels, 

# Obter a matrix de confusão com base nos conjuntos de teste.
# Ordem da matriz é de acordo com a ordem alfabética da pasta /data.
# 0 - Axe
# 1 - Banana
# 2 - Fork
# 3 - Moon
print('Confusion Matrix: ')
print(confusion_matrix(y_true, y_pred))


#Variaveis globais usadas para desenhar na tela.
drawing = False
last_x, last_y = None, None

def draw(event, x, y, flags, param) :
    """
        Função resposável por desenhar na tela
        ---

        Parametros:
        ---

            - event: Evento que está acontecendo na tela no momento
            - x: Posição X, onde está acontecendo o evento
            - y: Posição Y, onde está acontecendo o evento
            - flags, param: parametros default de qualquer função que usa o canvas no openCV
        
        Retorno:
        ---
            - Não retorna nada, pois trabalha com referencia.
    """
    global drawing, last_x, last_y

    if event == cv2.EVENT_LBUTTONDOWN :
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE :
        if drawing :
            if last_x == None or last_y == None :
                last_x = x
                last_y = y
            cv2.line(draw_img, (last_x, last_y), (x, y), (255,255,255), 5)
            last_x = x
            last_y = y
    elif event == cv2.EVENT_LBUTTONUP :
        drawing = False
        last_x = None
        last_y = None

def reshape_for_prediction(arr, w, h, d) :
    """
        Função que realiza o reshape da imagem para a predição da rede neural.

        Parametros:
        ---

            - arr: Matriz que contem o desenho feito no canvas do opencv.
            - w: Largura que a nova matriz terá.
            - h: Altura que a nova matriz terá.
            - d: profundiade que a nova matriz terá.
        
        Retorno:
        ---
            - Matriz redimencionada e convertida para uma imagem binária.
    """
    
    # Divide por 255 para tranformar os ranges de 0-255 para 0-1
    # Obs.: a transformação de 0-1 é opcional, mas facilita na hora da predição.
    return arr.reshape((1,d,w,h)).astype('float32')/255



# Instanciação dos canvas (Telas)
draw_img = np.zeros((300,300,3), np.uint8)
pred_img = np.zeros((100,300,3), np.uint8)

cv2.namedWindow('drawing')
cv2.namedWindow('prediction')
cv2.setMouseCallback('drawing', draw)
font = cv2.FONT_HERSHEY_SIMPLEX

while(1) :
    # exibe as imagens desenhadas nas telas.
    cv2.imshow('drawing', draw_img)
    cv2.imshow('prediction', pred_img)
    
     # copia a imagem da tela de desenho e utiliza ela para predição.
    img_to_pred = draw_img.copy()
    img_to_pred = cv2.cvtColor(img_to_pred, cv2.COLOR_BGR2GRAY) # converte ela de RGB para gray scale
    img_to_pred = cv2.resize(img_to_pred, (width,height)) # redimensiona a imagem para o tamanho width x height
    
    # redimenciona a imagem para o tamanho definido e faz um escalonamento de 0-255 para 0-1
    img_to_pred = reshape_for_prediction(img_to_pred, width, height, depth)
    
    # tenta predizer a imagem gerada.
    prediction = rna.predict(img_to_pred)
    prediction = np.argmax(prediction, axis=1)

    pred_img.fill(0)

    # Escreve o resultado da predição.
    if draw_img.any() :
        cv2.putText(pred_img, str(label_map[int(prediction)]), (10,70), font, 1.5, (255,255,255), 2, cv2.LINE_AA)

    #Captura a tecla após 20 ns
    k = cv2.waitKey(20)

    # 27 - ESC ou até que a tela seja fechada.
    if k == 27 or cv2.getWindowProperty('drawing', 0) == -1 or cv2.getWindowProperty('prediction', 0) == -1:
        cv2.destroyAllWindows()
        break
    elif k == ord('c') :
        draw_img.fill(0)
