import numpy as np
import zipfile
import gzip
import io
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import layers, models

from generate_fonts import generate_digital_dataset

def read_emnist_from_zip(path_to_zip):
    """
    Legge il file zip di EMNIST e restituisce due array numpy:
    - images: (N, 28, 28) uint8
    - labels: (N,) uint8
    """
    if not os.path.exists(path_to_zip):
        raise FileNotFoundError(f"File non trovato: {path_to_zip}")

    # Funzione interna per il parsing binario (IDX format)
    def parse_idx(content_gz):
        with gzip.GzipFile(fileobj=io.BytesIO(content_gz)) as f:
            data = f.read()
            
        magic = int.from_bytes(data[0:4], 'big')
        ndims = magic % 256

        if ndims == 1: # Labels
            # Offset 8 byte (4 magic + 4 size)
            return np.frombuffer(data, dtype=np.uint8, offset=8)
        elif ndims == 3: # Images
            # Offset 16 byte (4 magic + 4 size + 4 rows + 4 cols)
            return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        file_list = zf.namelist()
        
        # Cerca i file giusti dentro lo zip indipendentemente dalle cartelle
        img_file = next(n for n in file_list if "balanced-train-images" in n)
        lbl_file = next(n for n in file_list if "balanced-train-labels" in n)

        # Estrae i byte compressi e parsa
        with zf.open(img_file) as f:
            images = parse_idx(f.read())
            
        with zf.open(lbl_file) as f:
            labels = parse_idx(f.read())

    return images, labels



x_train, y_train = read_emnist_from_zip('assets/emnist.zip')
size = x_train.shape[0]
images = []

# Pre-Processig, ruoto le immagini di 90 gradi e le specchio per renderle "dritte"
for i in range(0, size):
    images.append(np.fliplr(np.rot90(x_train[i], -1)))
x_train = np.array(images)

# Aggiungo immagini di caratteri dei font create artificialmente
x_train_digital, y_train_digital = generate_digital_dataset(100000)

x_train = np.concatenate((x_train, x_train_digital), axis=0)
y_train = np.concatenate((y_train, y_train_digital), axis=0)

# Permuto le immagini
perm = np.random.permutation(len(x_train))
x_train = x_train[perm]
y_train = y_train[perm]

# Sistemo per TensorFlow
x_train.reshape(-1, 28, 28, 1) # aggiungo la dimensione del canale (scala di grigi)
x_train = x_train.astype('float32') / 255.0 # per trasformare i dati dal range 0-255 a 0-1 (per far lavorare la rete neurale)


# Creo il modello
model = keras.models.Sequential([
    keras.Input(shape=(28, 28)), # definisce la forma dell'input
    layers.Flatten(), # trasforma immagine da matrice 28x28 in un array di 784 elementi, necessario per i layers successivi.
    layers.Dense(128, activation='relu'), # prende in input i 784 numeri del layer precedente e crea 128 neuroni artificiali. 'relu' trasforma neuroni negativi in 0. Ritorna una lista di 128 numeri
    layers.Dense(47, activation='softmax') # prende in input i 128 numeri dell'hidden layer precedente, li collega a 47 neuroni finali (uno per ogni carattere EMNIST). 'softmax' trasforma i numeri in percentuali. Ritorna un vettore di possibilità
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
# categorical_crossentropy calcola la distanza matematica tra la distribuzione di probabilità predetta e quella reale. Utile quando ho più di 2 classi (in questo caso 47)

print("Inizio fase di addestramento...")
model.fit(x_train, y_train, epochs=10, verbose=2, validation_split=0.2)
print("Fine fase di addestramento.")
model.save('assets/model.keras')