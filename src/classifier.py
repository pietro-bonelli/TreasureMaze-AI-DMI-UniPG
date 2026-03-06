import numpy as np
import keras
import os

_model_path = 'assets/model.keras'

EMNIST_MAP = {
    0: '1',
    1: '2',
    2: '3',
    3: '4',
    4: 'S',
    5: 'T',
    6: 'X'
}

_model: keras.Model = None

# Carico il modello una sola volta per ottimizzare
def load():
    global _model, _model_path
    if _model is None:
        _model = keras.models.load_model(_model_path)
    return _model


def predict(image: np.ndarray):
    model = load()

    # Preparo l'immagine (reshape per ridimensionarla e converto da 0-255 a 0-1)
    image = image.reshape(1, 28, 28)
    image = image.astype('float32') / 255

    prediction = model.predict(image, verbose=0) # ottengo tutte le predizioni
    best_index = np.argmax(prediction)
    char = EMNIST_MAP.get(best_index, '?')

    return char