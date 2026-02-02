import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import plotImage, show_pipeline, orderPoints, sort_contours
from classifier import predict # Assumo che la tua funzione si chiami così o "predict"

image_path = "assets/dirty_maze.png"

def getBinaryImage(image):
    """
    FASE 1: PREPARAZIONE GLOBALE
    L'obiettivo qui è solo evidenziare i contorni della griglia per poterli trovare.
    Non ci preoccupiamo ancora del contenuto delle celle.
    """
    # plotImage(image, "Immagine originale")
    
    # 1. Scala di grigi (essenziale per l'analisi)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # plotImage(gray_image, "Scala di grigi")
    
    # 2. Rimozione rumore (Blur)
    blur = cv2.GaussianBlur(gray_image, (9, 9), 0) # Ho ridotto leggermente il kernel da 19 a 9 per non perdere dettagli angolari

    # 3. Threshold Adattivo (Ottimo per trovare linee in condizioni di luce variabile)
    # Uso GAUSSIAN_C perché gestisce meglio le ombre rispetto a MEAN_C
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    # plotImage(thresh, "Threshold Adattivo")

    # 4. "Chiusura" dei buchi (Morphology Close)
    # Serve a unire linee spezzate della griglia
    kernel = np.ones((5,5), np.uint8)
    thresh_morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Bordo di sicurezza
    # Disegno un rettangolo bianco intorno a tutto per chiudere eventuali griglie aperte ai lati
    h, w = thresh_morph.shape[:2]
    cv2.rectangle(thresh_morph, (0, 0), (w, h), 255, thickness=5)

    return thresh_morph, gray_image


import math
from scipy import ndimage

def preprocess_smart(img_gray):
    """
    Prende una cella grezza, trova il numero, lo ritaglia, 
    lo centra in 20x20 e poi in 28x28.
    """
    # 1. Threshold secco (Testo bianco su nero)
    # Usiamo 128 come spartiacque. Se è grigio scuro diventa nero.
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. Trova il "Bounding Box" del numero (ritaglia il vuoto attorno)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return thresh # Ritorna nero se non trova nulla
    x, y, w, h = cv2.boundingRect(coords)
    
    # Ritaglio effettivo del solo numero
    digit = thresh[y:y+h, x:x+w]
    
    # 3. Ridimensionamento "Safe" nel box 20x20
    # EMNIST vuole il carattere in un box 20x20, centrato in 28x28
    rows, cols = digit.shape
    
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        
    # Resize delicato (INTER_AREA è meglio per rimpicciolire)
    digit = cv2.resize(digit, (cols, rows), interpolation=cv2.INTER_AREA)
    
    # 4. Creazione tela nera 28x28
    padded = np.zeros((28, 28), dtype=np.uint8)
    
    # Calcolo coordinate per incollare il numero al centro geometrico
    pad_top = (28 - rows) // 2
    pad_left = (28 - cols) // 2
    
    padded[pad_top:pad_top+rows, pad_left:pad_left+cols] = digit
    
    # 5. Centramento tramite BARICENTRO (Center of Mass)
    # Questo è il segreto di MNIST/EMNIST. Spostiamo l'inchiostro al centro gravitazionale.
    cy, cx = ndimage.center_of_mass(padded)
    shiftx = np.round(14.0 - cx).astype(int)
    shifty = np.round(14.0 - cy).astype(int)
    
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    padded = cv2.warpAffine(padded, M, (28, 28))
    
    # 6. Ultimo tocco: Ingrassare e Sfocare
    kernel = np.ones((2, 2), np.uint8)
    padded = cv2.dilate(padded, kernel, iterations=1)
    # Un blur leggerissimo ammorbidisce i pixel digitali
    padded = cv2.GaussianBlur(padded, (3, 3), 0)
    
    return padded


def findBoxes(binaryImage, originalImage, grayImage):
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursImg = originalImage.copy()
    
    validContours = [] 
    croppedImages = [] # Qui salveremo le immagini pronte per l'IA
    
    # Filtro i contorni per trovare solo i quadrati delle celle
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area < 1000 or area > 50000):
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.02 * perimeter 
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4: 
            validContours.append(approx)

    # Ordinamento delle celle (alto-sx -> basso-dx)
    if len(validContours) > 0:
        validContours = sort_contours(validContours) 
    else:
        print("Nessuna cella trovata")
        return []

    # Estrazione e Processing di ogni singola cella
    for i, box in enumerate(validContours):
            # ... (parte del drawContours identica) ...
            
            # PERSPECTIVE TRANSFORM
            rect = orderPoints(box)
            dst_pts = np.array([[0, 0], [28, 0], [28, 28], [0, 28]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst_pts)
            
            # IMPORTANTE: Warpa l'immagine GRIGIA (grayImage), non quella binaria
            raw_cell = cv2.warpPerspective(grayImage, M, (28, 28))
            
            # CHIAMA LA NUOVA FUNZIONE
            final_cell = preprocess_smart(raw_cell)
            
            croppedImages.append(final_cell)
    
    # Visualizzazione per debug
    debug_steps = [(contoursImg, "Griglia Individuata")]
    # Mostriamo le prime 16 celle per vedere se sono "grasse" e leggibili
    for i in range(min(len(croppedImages), 16)): 
        debug_steps.append((croppedImages[i], f"Cella {i}"))
        
    show_pipeline(debug_steps)
    
    return croppedImages


# --- MAIN ---

image = cv2.imread(image_path)
if image is None:
    print(f"Errore: Impossibile caricare {image_path}")
    exit()

# 1. Trova la griglia
binary_grid_map, gray_full = getBinaryImage(image)

# 2. Estrai e processa le celle
boxes = findBoxes(binary_grid_map, image, gray_full)

print(f"Trovate {len(boxes)} celle.")

# 3. Predizione
labirinto_str = []
print("\n--- RISULTATI PREDITTI ---")
for i, box in enumerate(boxes):
    # La box è già preprocessata (grassa e sfocata), pronta per predict
    prediction = predict(box) # Usa il nome corretto della tua funzione importata
    labirinto_str.append(prediction)
    print(f"Cella {i}: {prediction}")

# Se vuoi vederli tutti in fila:
# print(labirinto_str)