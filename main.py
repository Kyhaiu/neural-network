import cv2
import os
import numpy as np
import keras.backend as kb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


path = './data'

# Dataset final
xtotal = []
ytotal = []

#Lista de aquivos no diretório /data
filenames = os.listdir(path)

# O Dataset terá no max 200000 elementos.
# O numero de max de cada elementos de um conjunto será a média no max.
slice_train = int(200000/len(filenames))

# Seed usada para embaralhar os dados antes de dividir
seed = np.random.randint(1, 10e7)

# Mapa de labels dos conjuntos
label_map = {} # maps integer values of labels to their corresponding strings

i = 0
for fname in filenames :
    
    # Caso o conjunto não seja do formato proprio para o np.load, ele só ignora.
    if '.npy' not in fname :
        continue

    x = np.load(path + '/' +fname)
    
    # realiza uma escala nos dados de 0-255 para 0-1
    x = x.astype('float32') / 255 
    label_name = fname.split('.npy')[0]

    # Replica obtida para em um vetor com o mesmo numero de elementos que tem em x.
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

# Realiza o reshape dos conjuntos de teste.
# Todos os desenhos simplificados, foram rendenizados em uma matrix 28x28 em grayscale bitmap no formato .npy
# Código aplicado na geração dos arquivos https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

encoder = LabelEncoder()
encoded_y_train = encoder.fit_transform(y_train)
y_train_categorical = to_categorical(encoded_y_train)

encoded_y_test = encoder.fit_transform(y_test)
y_test_categorical = to_categorical(encoded_y_test)


def create_rna(num_classes, input_shape):
    rna = Sequential()
    rna.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    rna.add(MaxPooling2D())
    rna.add(Conv2D(15, (3, 3), activation='relu'))
    rna.add(MaxPooling2D())
    rna.add(Dropout(0.2))
    rna.add(Flatten())
    rna.add(Dense(128, activation='relu'))
    rna.add(Dense(50, activation='relu'))
    rna.add(Dense(num_classes, activation='softmax'))

    rna.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return rna

# Número de classes
number_class = len(label_map)

# Tamanho das imagens do dataset
width = height = 28

depth = 1

if kb.image_data_format == 'channels_first':
    input_shape = (depth, width, height)
    train_features = x_train.reshape(x_train.shape[0], depth, width, height).astype('float32')
    test_features = x_test.reshape(x_test.shape[0], depth, width, height).astype('float32')
else:
    input_shape = (width, height, depth)
    train_features = x_train.reshape(x_train.shape[0], width, height, depth).astype('float32')
    test_features = x_test.reshape(x_test.shape[0], width, height, depth).astype('float32')

train_labels = y_train_categorical
test_labels = y_test_categorical

try:
    print("Carregando RNA...")
    rna = load_model('./Model/rede_neural.tf')
except:
    print("Não existe RNA, Criando uma nova...")

    rna = create_rna(number_class, input_shape)
    rna.fit(train_features, train_labels, batch_size=100, epochs=3)
    rna.save('./Model/rede_neural.tf')

eval_results = rna.evaluate(test_features, test_labels)
print('Loss: ', eval_results[0], 'Accuracy: ', eval_results[1])


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
    # reshape to (num_of_samples, width, height, depth (or channels)) or (num_of_samples, depth (or channels) width, height)
    # according to the configuration
    # and convert from 0-255 range to 0-1 range
    if kb.image_data_format == 'channels_first' :
        return arr.reshape((1,d,w,h)).astype('float32') / 255
    else :
        return arr.reshape((1,w,h,d)).astype('float32') / 255



# Instanciação dos canvas (Telas)
draw_img = np.zeros((300,300,3), np.uint8)
pred_img = np.zeros((100,300,3), np.uint8)

cv2.namedWindow('drawing')
cv2.namedWindow('prediction')
cv2.setMouseCallback('drawing', draw)
font = cv2.FONT_HERSHEY_SIMPLEX

while(1) :
    # show images on their windows
    cv2.imshow('drawing', draw_img)
    cv2.imshow('prediction', pred_img)
    
     # copy the image on the 'drawing' window and use it to predict
    img_to_pred = draw_img.copy()
    img_to_pred = cv2.cvtColor(img_to_pred, cv2.COLOR_BGR2GRAY) # convert BGR to grayscale image
    img_to_pred = cv2.resize(img_to_pred, (width,height)) # resize to a width x height image
    
    # reshape to a suitable shape for the model and convert values from 0-255 range to 0-1 range
    img_to_pred = reshape_for_prediction(img_to_pred, width, height, depth)
    
    # predict the image
    prediction = rna.predict(img_to_pred)
    prediction = np.argmax(prediction, axis=1)

    pred_img.fill(0)


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
