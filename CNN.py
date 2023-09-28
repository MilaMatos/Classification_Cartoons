import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

# Defina uma variável global para armazenar o modelo treinado
global model
model = None

# Obtenha o diretório atual onde seu arquivo .py está localizado
script_dir = os.path.dirname(__file__)

# Define o diretório onde suas imagens estão localizadas
data_dir = os.path.join(script_dir, 'TEST')

# Use ImageDataGenerator para carregar os dados
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)  # Normalização

def preprocess_image(img_path):
    # Carregue a imagem original
    imagem = cv2.imread(img_path)

    # Verifique se a imagem foi carregada com sucesso
    if imagem is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return None  # Retorne None para indicar que o processamento falhou

    # Redimensionando a imagem
    imagem_redimensionada = cv2.resize(imagem, (224, 224))

    # Aumento de contraste
    imagem_em_escala_de_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)
    #imagem_em_escala_de_cinza = cv2.equalizeHist(imagem_em_escala_de_cinza)

    # Retorne a imagem processada
    return imagem_em_escala_de_cinza  # Mantenha a matriz 2D em escala de cinza com aumento de contraste


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
labels = np.unique(image_labels)

# Defina o número de folds
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Variáveis para armazenar resultados
all_train_histories = []
all_val_histories = []

# Defina o valor de batch_size
batch_size = 32  # Você pode ajustar esse valor conforme necessário


def treino():
    global model  # Use a variável global

    # Loop através dos folds
    for train_index, val_index in kf.split(image_data):
        train_data = image_data[train_index]
        train_labels = image_labels[train_index]
        val_data = image_data[val_index]
        val_labels = image_labels[val_index]

        # Pré-processar as imagens de treinamento e validação
        train_data_processed = [preprocess_image(img_path) for img_path in train_data]
        val_data_processed = [preprocess_image(img_path) for img_path in val_data]

        # Remova entradas None (imagens que não puderam ser carregadas)
        train_data_processed = [img for img in train_data_processed if img is not None]
        val_data_processed = [img for img in val_data_processed if img is not None]

        # Converter as imagens em listas de matrizes 1D
        train_data_processed = [img.flatten() for img in train_data_processed]
        val_data_processed = [img.flatten() for img in val_data_processed]

        # Certifique-se de que todas as imagens processadas tenham o mesmo tamanho
        train_data_processed = [img for img in train_data_processed if len(img) == len(train_data_processed[0])]
        val_data_processed = [img for img in val_data_processed if len(img) == len(val_data_processed[0])]

        # Converta as listas em arrays numpy
        train_data_processed = np.array(train_data_processed)
        val_data_processed = np.array(val_data_processed)

        train_df = pd.DataFrame({'filename': train_data, 'class': train_labels})
        val_df = pd.DataFrame({'filename': val_data, 'class': val_labels})

        train_generator = datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='filename',
            y_col='class',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_generator = datagen.flow_from_dataframe(
            dataframe=val_df,
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

        model.compile(optimizer='rmsprop',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

        # Treine o modelo por 1 época (ou o número de épocas desejado)
        history = model.fit(train_generator, epochs=1, validation_data=val_generator)
        all_train_histories.append(history.history['accuracy'])
        all_val_histories.append(history.history['val_accuracy'])

    # Avalie o desempenho médio no conjunto de validação
    average_val_accuracy = np.mean([history[-1] for history in all_val_histories])
    print(f'Average Validation Accuracy Across Folds: {average_val_accuracy}')

    # Salve o modelo treinado para uso posterior
    model.save('modelo_treinado.h5')

    return model

# Função para carregar o modelo treinado ou treiná-lo se necessário
def carregar_ou_treinar_modelo():
    global model
    if model is None:
        if os.path.exists('modelo_treinado.h5'):
            model = tf.keras.models.load_model('modelo_treinado.h5')
        else:
            model = treino()
    return model

def preprocess_image_for_prediction(img_path):
    # Carregue a imagem original
    imagem = cv2.imread(img_path)

    # Verifique se a imagem foi carregada com sucesso
    if imagem is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return None  # Retorne None para indicar que o processamento falhou

    # Redimensionando a imagem
    imagem_redimensionada = cv2.resize(imagem, (224, 224))

    # Aumento de contraste
    imagem_em_escala_de_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)
    #imagem_em_escala_de_cinza = cv2.equalizeHist(imagem_em_escala_de_cinza)

    # Expanda as dimensões para corresponder à forma esperada do modelo (1, 224, 224, 3)
    imagem_processada = np.expand_dims(imagem_em_escala_de_cinza, axis=0)
    imagem_processada = np.stack((imagem_processada,) * 3, axis=-1)  # Replica o canal em todos os três canais
    imagem_processada = np.expand_dims(imagem_processada, axis=-1)
    
    # Normalizar a imagem
    imagem_processada = imagem_processada / 255.0

    return imagem_processada


# Função para fazer a classificação da imagem
def classificar_imagem(imagem_path):
    model = carregar_ou_treinar_modelo()  # Carrega ou treina o modelo
    # Pré-processamento da imagem
    imagem_processada = preprocess_image_for_prediction(imagem_path)

    if imagem_processada is None:
        return "Falha no pré-processamento", None

    # Faça a previsão usando o modelo
    previsao = model.predict(imagem_processada)

    # Obtenha a classe prevista com a maior probabilidade
    classe_prevista = np.argmax(previsao)

    # Imprima as classes previstas e suas probabilidades
    print(f"\n\nResultado da classificação para a imagem: {imagem_path}")
    print("\nClassificação das 3 classes mais prováveis:")
    top_classes_indices = np.argsort(previsao[0])[::-1][:3]
    top_classes_names = [labels[i] for i in top_classes_indices]
    top_classes_probs = [previsao[0][i] for i in top_classes_indices]
    for i, (classe, probabilidade) in enumerate(zip(top_classes_names, top_classes_probs), 1):
        print(f"{i}. Classe: {classe}, Probabilidade: {probabilidade:.4f}")

    # Imprima todas as classes e suas probabilidades
    print("\nProbabilidades para todas as classes:")
    for i, (classe, probabilidade) in enumerate(zip(labels, previsao[0]), 1):
        print(f"{i}. Classe: {classe}, Probabilidade: {probabilidade:.4f}")

    return labels[classe_prevista], previsao[0][classe_prevista]


# Insira o caminho da imagem que você deseja classificar
imagem_path = os.path.join(script_dir, 'teste_ML_3.jpg')
# Classifique a imagem
classe_prevista = classificar_imagem(imagem_path)


#FAZER PROGRAMA PARA SEPARAR OS FRAMES DE ACORDO COM A SIMILARIDADE PARA REDUZIR A BASE DE DADOS
#FAZER VERSÕES COM OUTROS CLASSIFICADORES/CAMADAS PARA COMPARAR

"""Teste estatístico:
    Ver se existem outros códigos para comparar? 
    

"""
