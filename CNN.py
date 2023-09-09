'''

import matplotlib.pyplot as plt
import tensorflow as tf
#from tf.keras.preprocessing.image import ImageDataGenerator
#from tf.keras import layers, models
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# Define o diretório onde suas imagens estão localizadas
data_dir = 'E:\zzzzzanime\Images_ML'

# Use ImageDataGenerator para carregar os dados
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)  # Normalização

# Carregue os dados em um array numpy
image_data = []
image_labels = []

for subdir in os.listdir(data_dir):
    sub_dir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(sub_dir_path):
        for img_file in os.listdir(sub_dir_path):
            img_path = os.path.join(sub_dir_path, img_file)
            image_data.append(img_path)
            image_labels.append(subdir)

image_data = np.array(image_data)
image_labels = np.array(image_labels)

# Defina o número de folds
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Variáveis para armazenar resultados
all_train_histories = []
all_val_histories = []

# Defina o valor de batch_size
batch_size = 32  # Você pode ajustar esse valor conforme necessário


def treino():
    # Loop através dos folds
    for train_index, val_index in kf.split(image_data):
        train_data = image_data[train_index]
        train_labels = image_labels[train_index]
        val_data = image_data[val_index]
        val_labels = image_labels[val_index]

        train_generator = datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': train_data, 'class': train_labels}),
            x_col='filename',
            y_col='class',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_generator = datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': val_data, 'class': val_labels}),
            x_col='filename',
            y_col='class',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Crie e compile o modelo (certifique-se de usar o número correto de classes)
        num_classes = len(np.unique(train_labels))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # Use o número correto de unidades

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

        # Treine o modelo por 1 época (ou o número de épocas desejado)
        history = model.fit(train_generator, epochs=1, validation_data=val_generator)
        all_train_histories.append(history.history['accuracy'])
        all_val_histories.append(history.history['val_accuracy'])

    # Avalie o desempenho médio no conjunto de validação
    average_val_accuracy = np.mean([history[-1] for history in all_val_histories])
    print(f'Average Validation Accuracy Across Folds: {average_val_accuracy}')
    
    return model

model = treino()


import cv2 
import numpy as np

# Função para fazer a classificação
def classificar_imagem(imagem_path):
    # Carregue a imagem
    imagem = cv2.imread(imagem_path)
    # Redimensione a imagem para o tamanho de entrada do modelo (por exemplo, 224x224)
    imagem = cv2.resize(imagem, (224, 224))
    # Pré-processamento da imagem (normalização, expansão de dimensão)
    imagem = np.expand_dims(imagem, axis=0) / 255.0

    # Faça a previsão usando o modelo
    previsao = model.predict(imagem)

    print(previsao)

    # Obtenha a classe prevista
    classe_prevista = np.argmax(previsao, axis=1)  # Supondo classificação em one-hot encoding

    # Mapeie o índice da classe prevista para o nome da classe
    nome_classe_prevista =image_labels[classe_prevista[0]]

    print('\n\n',classe_prevista)

    return nome_classe_prevista

# Insira o caminho da imagem que você deseja classificar
imagem_path = 'Images_ML\Futurama/100.jpg'

# Classifique a imagem
classe_prevista = classificar_imagem(imagem_path)


print('\n\nLabels: ', np.unique(image_labels))

'''