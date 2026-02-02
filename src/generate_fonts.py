import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import cv2

# Mappa per convertire i caratteri in label (0-46) come EMNIST
# Assicuriamoci che corrisponda alla tua mappa EMNIST
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt" 
# Nota: EMNIST 'balanced' ha 47 classi. Se usi solo maiuscole e numeri, riduciamo.

def get_label_from_char(char):
    # Ritorna l'indice basato sulla mappa EMNIST standard
    if char.isdigit():
        return int(char)
    elif char.isupper():
        return ord(char) - 55 # A=10, B=11...
    else:
        # Gestione minuscole mappate in coda (se servono)
        # Per ora semplifichiamo: se il tuo labirinto ha solo maiuscole
        return ord(char.upper()) - 55

def generate_digital_dataset(num_samples=2000):
    """
    Genera un dataset di immagini 28x28 con font digitali.
    """
    dataset_images = []
    dataset_labels = []

    # Lista di font comuni (assicurati di averli o metti path assoluti)
    # Su Windows di solito sono in C:/Windows/Fonts/
    font_paths = [
        "arial.ttf", 
        "cour.ttf",   # Courier (simile al tuo labirinto)
        "tahoma.ttf",
        "verdana.ttf",
        "consola.ttf" # Consolas
    ]

    print(f"Generazione di {num_samples} immagini sintetiche...")

    for _ in range(num_samples):
        # 1. Scelta casuale carattere e font
        char_to_draw = random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") # Solo maiuscole e numeri
        font_name = random.choice(font_paths)
        
        try:
            # Dimensione variabile per robustezza (tra 18 e 22)
            font_size = random.randint(18, 22)
            font = ImageFont.truetype(font_name, font_size)
        except IOError:
            # Se non trova il font, usa quello di default
            font = ImageFont.load_default()

        # 2. Creazione immagine nera
        img = Image.new('L', (28, 28), color=0) # 'L' = scala di grigi, 0 = nero
        draw = ImageDraw.Draw(img)

        # 3. Centratura del testo
        # Calcoliamo la bounding box del testo per centrarlo
        bbox = draw.textbbox((0, 0), char_to_draw, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        x = (28 - text_w) / 2
        y = (28 - text_h) / 2 - 2 # Un leggero offset verticale spesso aiuta

        # 4. Disegno testo bianco (255)
        draw.text((x, y), char_to_draw, font=font, fill=255)

        # 5. Conversione in numpy array
        np_img = np.array(img)
        
        # Opzionale: Aggiungi un po' di rumore o blur per simulare lo screenshot sporco
        # np_img = cv2.GaussianBlur(np_img, (3,3), 0)

        dataset_images.append(np_img)
        dataset_labels.append(get_label_from_char(char_to_draw))

    return np.array(dataset_images), np.array(dataset_labels)

# Test rapido se lanci questo file
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, y = generate_digital_dataset(10)
    plt.imshow(x[0], cmap='gray')
    plt.title(f"Label: {y[0]}")
    plt.show()