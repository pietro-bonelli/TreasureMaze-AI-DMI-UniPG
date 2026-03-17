import numpy as np
import zipfile
import gzip
import io
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping

#from TreasureMaze.old.generate_fonts import generate_digital_dataset

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
# Mantengo solo i caratteri che mi interessano (1-4, S, T, X)
labels_to_keep = [1, 2, 3, 4, 28, 29, 33]
# Filtro gli array
mask = np.isin(y_train, labels_to_keep)
x_train = x_train[mask]
y_train = y_train[mask]
print(f"Dataset ridotto a {len(y_train)} immagini.")
# Rimappo le etichette da 0 a 7 (altrimenti la rete neurale si aspetta 33 oggetti al posto di 7)
label_map = {
    1: 0, # 1
    2: 1, # 2
    3: 2, # 3
    4: 3, # 4
    28: 4, # S
    29: 5, # T
    33: 6 # X
}
y_train = np.array([label_map[y] for y in y_train]) # applica la nuova mappa a y_train.


size = x_train.shape[0]
images = []

# Pre-Processig, ruoto le immagini di 90 gradi e le specchio per renderle "dritte"
for i in range(0, size):
    images.append(np.fliplr(np.rot90(x_train[i], -1)))
x_train = np.array(images)

# Aggiungo immagini di caratteri dei font create artificialmente
#x_train_digital, y_train_digital = generate_digital_dataset(100)

#x_train = np.concatenate((x_train, x_train_digital), axis=0)
#y_train = np.concatenate((y_train, y_train_digital), axis=0)

# Permuto le immagini
perm = np.random.permutation(len(x_train))
x_train = x_train[perm]
y_train = y_train[perm]

# Sistemo per TensorFlow
x_train.reshape(-1, 28, 28, 1) # aggiungo la dimensione del canale (scala di grigi)
x_train = x_train.astype('float32') / 255.0 # per trasformare i dati dal range 0-255 a 0-1 (per far lavorare la rete neurale)


# Creo il modello
model = keras.models.Sequential([
    keras.Input(shape=(28, 28, 1)),
    # Data Augmentation: applico piccole trasformazioni alle immagini (rotazioni/zoom) per addestrare meglio la rete a riconoscere i caratteri correttamente.
    layers.RandomRotation(factor=0.03, fill_mode='constant', fill_value=0.0), # max 15% rotazione
    layers.RandomZoom(height_factor=0.05, width_factor=0.1, fill_mode='constant', fill_value=0.0), # zoom in/out max 10%
    # Parte 1 CNN - Riconosce tratti base (come linee verticali/orizzontali)
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)), # Rimpicciolisce l'immagine senza perdere caratteristiche importanti.
    # Parte 2 CNN - Combino i tratti per capire forme complesse (unisce i tratti base)
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), # kernel_size = quanti pixel guardare alla volta
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Rete densa per la decisione finale
    layers.Flatten(), # crea un array monodimensionale
    layers.Dropout(0.5), # spegne a caso il 50% dei neuroni (per evitare overfitting)
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax') # le 7 classi finali (1, 2, 3, 4, S, T, X)
])
# Relu trasforma i risultati negativi in 0 (funzione di attivazione)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
# categorical_crossentropy calcola la distanza matematica tra la distribuzione di probabilità predetta e quella reale. Utile quando ho più di 2 classi (in questo caso 7)

# Implemento Early Stopping per fermare addestramento al momento ottimale ed evitare overfitting.
# Posso implementarla poiché ho inserito un validation test con il quale fare il confronto (rispetto al solo training)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience = 2, # Se var_loss non scende per 2 epoche di fila, innesca l'Early Stopping.
    restore_best_weights=True, # riporta la rete ai pesi dell'epoca migliore.
    mode='min' # per minimizzare la var_loss 
)

print("Inizio fase di addestramento...")
model.fit(x_train, y_train, epochs=15, verbose=2, validation_split=0.2, callbacks=[early_stopping])
print("Fine fase di addestramento.")
model.save('assets/model.keras')