import numpy as np
import matplotlib.pyplot as plt
import random
import zipfile
import gzip
import io
import os

# Importiamo la tua funzione di predizione
from classifier import predict

def read_zip_complete(path_to_zip):
    print(f"Lettura completa da {path_to_zip}...")
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        file_list = zf.namelist()
        img_file = next(n for n in file_list if "balanced-train-images" in n)
        lbl_file = next(n for n in file_list if "balanced-train-labels" in n)
        
        # Immagini
        with zf.open(img_file) as f:
            with gzip.GzipFile(fileobj=io.BytesIO(f.read())) as gz:
                images = np.frombuffer(gz.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        
        # Etichette
        with zf.open(lbl_file) as f:
            with gzip.GzipFile(fileobj=io.BytesIO(f.read())) as gz:
                labels = np.frombuffer(gz.read(), dtype=np.uint8, offset=8)
                
    return images, labels

# --- MAIN ---
zip_path = os.path.join("assets", "emnist.zip")
images, labels = read_zip_complete(zip_path)

print("Cerco un numero nel dataset per testare la Rete...")

# Continua a pescare a caso finché non trova un indice che corrisponde a un numero (Label < 10)
while True:
    idx = random.randint(0, len(images) - 1)
    real_label = labels[idx]
    
    if real_label < 10: # Trovato un numero!
        print(f"Trovato! Indice {idx}, è il numero reale: {real_label}")
        
        # Prepara l'immagine
        sample_raw = images[idx]
        sample_ready = np.fliplr(np.rot90(sample_raw, k=-1))
        
        # Chiedi alla rete
        prediction = predict(sample_ready)
        
        print(f"La Rete Neurale dice: {prediction}")
        
        plt.imshow(sample_ready, cmap='gray')
        plt.title(f"Reale: {real_label} -> AI: {prediction}")
        plt.show()
        break