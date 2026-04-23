import numpy as np
import zipfile
import gzip
import io
import os
import matplotlib.pyplot as plt

def read_emnist_from_zip(path_to_zip):
    """
    Legge il file zip di EMNIST e restituisce due array numpy:
    - images: (N, 28, 28) uint8
    - labels: (N,) uint8
    """
    if not os.path.exists(path_to_zip):
        raise FileNotFoundError(f"File non trovato: {path_to_zip}")

    def parse_idx(content_gz):
        with gzip.GzipFile(fileobj=io.BytesIO(content_gz)) as f:
            data = f.read()
            
        magic = int.from_bytes(data[0:4], 'big')
        ndims = magic % 256

        if ndims == 1: 
            return np.frombuffer(data, dtype=np.uint8, offset=8)
        elif ndims == 3: 
            return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        file_list = zf.namelist()
        img_file = next(n for n in file_list if "balanced-train-images" in n)
        lbl_file = next(n for n in file_list if "balanced-train-labels" in n)

        with zf.open(img_file) as f:
            images = parse_idx(f.read())
        with zf.open(lbl_file) as f:
            labels = parse_idx(f.read())

    return images, labels

if __name__ == "__main__":
    print("Caricamento dataset EMNIST...")
    # Assicurati che il path sia corretto rispetto a dove lanci lo script
    x_train, y_train = read_emnist_from_zip('assets/emnist.zip')

    # Filtriamo SOLO per la label 4 originale di EMNIST
    mask_4 = (y_train == 4)
    x_fours = x_train[mask_4]
    
    print(f"Trovati {len(x_fours)} campioni del numero '4'.")

    # Pre-Processing: applichiamo la stessa rotazione e flip che usi nel main
    processed_fours = []
    for img in x_fours:
        processed_fours.append(np.fliplr(np.rot90(img, -1)))
    processed_fours = np.array(processed_fours)

    # Impostiamo una griglia 10x10 per vedere 100 campioni
    NUM_ROWS = 10
    NUM_COLS = 10
    TOTAL_SAMPLES = NUM_ROWS * NUM_COLS

    # Selezioniamo 100 indici casuali
    np.random.seed() # Assicura casualità ad ogni esecuzione
    random_indices = np.random.choice(len(processed_fours), TOTAL_SAMPLES, replace=False)
    samples = processed_fours[random_indices]

    # Stampa a video: figsize(10, 10) mantiene la proporzione quadrata
    fig, axes = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(10, 10))
    fig.suptitle("100 varianti di '4' nel dataset EMNIST", fontsize=16, fontweight='bold', y=0.95)
    
    for i, ax in enumerate(axes.flat):
        # Mappiamo in scala di grigi standard (nero su bianco per vederli meglio)
        ax.imshow(samples[i], cmap='gray_r')
        ax.axis('off') # Nascondiamo gli assi

    # Riduciamo drasticamente lo spazio bianco tra una riga/colonna e l'altra
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    plt.show()