import numpy as np
import keras
import os

_model_path = 'assets/model.keras'

EMNIST_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

ALLOWED_CHARS = ['1', '2', '3', '4', 'S', 'T', 'X']

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

    predictions = model.predict(image, verbose=0)[0] # ottengo tutte le predizioni
    best_char = '?'
    best_percentage = -1 # inizializzo

    for index, score in enumerate(predictions):
        char = EMNIST_MAP.get(index, '?')
        if char in ALLOWED_CHARS:
            if score > best_percentage:
                best_percentage = score
                best_char = char

    """# Effettuo la predizione
    prediction = model.predict(image, verbose=0)
    best_index = np.argmax(prediction)
    char = EMNIST_MAP.get(best_index, '?')"""
    return best_char