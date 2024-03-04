import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Função para carregar os dados de áudio e os rótulos
def load_audio_data_and_labels(data_dir, input_shape):
    audio_data = []
    labels = []
    classes = os.listdir(data_dir)
    print("Classes encontradas:", classes)
    for class_dir in classes:
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for audio_file in os.listdir(class_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(class_path, audio_file)
                    print("Carregando arquivo:", file_path)
                    audio, sr = librosa.load(file_path, sr=None)
                    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
                    if spectrogram.shape[1] < input_shape[1]:
                        padding = input_shape[1] - spectrogram.shape[1]
                        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
                    else:
                        spectrogram = spectrogram[:, :input_shape[1]]
                    audio_data.append(spectrogram)
                    labels.append(class_dir)
    label_to_numeric = {label: index for index, label in enumerate(np.unique(labels))}
    labels = [label_to_numeric[label] for label in labels]
    return np.array(audio_data), np.array(labels), label_to_numeric

# Função para construir o modelo CNN
def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Definir o formato desejado para os espectrogramas
n_frames = 128
n_features = 128
input_shape = (n_frames, n_features, 1)

# Carregar dados
data_dir = 'data'
train_images, train_labels, label_to_numeric = load_audio_data_and_labels(data_dir, input_shape)

# Dividir os dados em conjunto de treinamento e teste
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Adicionar uma dimensão para canais de entrada
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Dimensões dos dados de entrada
input_shape = train_images.shape[1:]
num_classes = len(np.unique(train_labels))

# Construir modelo
model = build_model(input_shape, num_classes)

# Compilar modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar modelo
print("Iniciando treinamento do modelo...")
model.fit(train_images, train_labels, epochs=50, batch_size=32, validation_data=(test_images, test_labels))

# Avaliar modelo
print("Avaliando o modelo...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)

# Salvar modelo treinado
print("Salvando o modelo treinado...")
try:
    model.save('audio_model.keras')  # Salvando como .keras
    print("Modelo treinado salvo com sucesso!")
except Exception as e:
    print("Ocorreu um erro ao salvar o modelo treinado:", e)
