import cv2
import numpy as np
import os

# Carica l'immagine pulita
img_path = 'assets/test_maze.png'
if not os.path.exists(img_path):
    print("Devi prima generare l'immagine pulita!")
    exit()

img = cv2.imread(img_path)

# 1. Aggiungiamo rumore (Noise)
# Generiamo rumore casuale (distribuzione gaussiana)
row, col, ch = img.shape
mean = 0
var = 0.5
sigma = var**0.5
gauss = np.random.normal(mean, sigma, (row, col, ch))
gauss = gauss.reshape(row, col, ch)
# Aggiungiamo il rumore all'immagine
noisy_img = img + gauss * 50 # Moltiplico per aumentare l'effetto

# Normalizziamo per rimanere nel range 0-255
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# 2. Sfocatura (Blur)
# Simula una foto mossa o fuori fuoco
dirty_img = cv2.GaussianBlur(noisy_img, (5, 5), 0)

# 3. Ombreggiatura (Illuminazione non uniforme)
# Creiamo una sfumatura scura
mask = np.zeros((row, col), dtype=np.uint8)
for i in range(row):
    # Crea un gradiente che scurisce verso il basso
    val = 255 - (i / row * 100)
    mask[i, :] = val
# Uniamo la maschera ai 3 canali
shadow_mask = cv2.merge([mask, mask, mask])
# Applichiamo l'ombra
dirty_img = cv2.bitwise_and(dirty_img, shadow_mask)

# Salva
output_path = 'assets/dirty_maze.png'
cv2.imwrite(output_path, dirty_img)
print(f"Immagine sporca creata: {output_path}")