import numpy as np
from tensorflow.keras.models import load_model
import librosa

# Carregar modelo treinado
model = load_model('models/audio_model.keras')

# Mapeamento de números de classe para nomes de classe
label_to_numeric = {0: '10', 1: '20', 2: 'A', 3: 'B'}  # Atualize conforme necessário

# Função para fazer previsões em novos dados de áudio
def predict_audio(audio_file_path, segment_length=2):
    audio, sr = librosa.load(audio_file_path, sr=None)
    segment_samples = sr * segment_length  # Número de amostras por segmento
    num_segments = len(audio) // segment_samples  # Número total de segmentos
    classes_found = []  # Lista para armazenar as classes encontradas

    # Iterar sobre os segmentos e fazer previsões
    for i in range(num_segments):
        segment = audio[i * segment_samples: (i + 1) * segment_samples]  # Obter o segmento atual
        spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
        input_shape = model.input_shape[1:]  # Obtendo o formato de entrada do modelo
        if spectrogram.shape[1] < input_shape[1]:
            padding = input_shape[1] - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant')
        else:
            spectrogram = spectrogram[:, :input_shape[1]]
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Adicionando dimensão de lote
        prediction = model.predict(spectrogram)
        predicted_class = np.argmax(prediction)
        class_name = label_to_numeric[predicted_class]
        classes_found.append(class_name)  # Adicionar classe encontrada à lista
    
    return classes_found