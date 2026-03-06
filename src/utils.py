# Funzione per plot corretto delle immagini
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def plotImage(image, title, saveToFile = False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # per evitare colori sballati
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    if saveToFile:
        plt.savefig('plots/{title}.png')
    plt.show()

def show_pipeline(images_with_titles, cols=10):
    """
    Mostra una lista di tuple (immagine, titolo) in una griglia dinamica.
    cols: numero di immagini per riga (10 è perfetto per il tuo labirinto)
    """
    n = len(images_with_titles)
    if n == 0:
        return

    # Calcola il numero di righe necessarie (arrotondamento per eccesso)
    rows = math.ceil(n / cols)

    # Crea la figura dinamicamente. Moltiplichiamo per 2 o 3 pollici a cella
    # per avere una finestra sufficientemente grande.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Matplotlib restituisce 'axes' in formati diversi a seconda delle dimensioni.
    # .flatten() lo trasforma in un array 1D comodo da iterare.
    if n == 1:
        axes = [axes]
    elif rows > 1 or cols > 1:
        axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            img, title = images_with_titles[i]
            
            # Matplotlib vuole RGB, OpenCV usa BGR. Se l'immagine è a colori, convertiamo.
            # Se è scala di grigi (2 dimensioni), usiamo cmap='gray'
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            ax.set_title(title, fontsize=8)
        
        # Nascondiamo le coordinate (numeri sugli assi) per non fare confusione
        ax.axis('off') 

    # Compatta il layout per evitare che i titoli si sovrappongano alle foto
    plt.tight_layout()
    plt.show()


def orderPoints(points):
    points = points.reshape((4, 2))
    rect = np.zeros((4,2), dtype="float32")
    
    sum = np.sum(points, axis=1)
    rect[0] = points[np.argmin(sum)] # Top-left
    rect[2] = points[np.argmax(sum)] # Bottom-Right

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)] # Top-Right
    rect[3] = points[np.argmax(diff)] # Bottom-Left
    
    return rect

def sort_contours(contours):
    """
    Ordina i contorni (Top-Bottom, poi Left-Right) senza conoscere 
    il numero di colonne a priori.
    """
    # 1. Calcoliamo i Bounding Box per tutti
    # box = (x, y, w, h)
    boxes = [cv2.boundingRect(c) for c in contours]
    
    # 2. Uniamo contorni e box e ordiniamo TUTTO per Y (Alto -> Basso)
    #    Questo ci serve per processarli riga per riga
    zipped = list(zip(contours, boxes))
    zipped.sort(key=lambda b: b[1][1]) # b[1][1] è la coordinata Y
    
    if not zipped:
        return []

    final_contours = []
    
    # Inizializziamo la prima riga col primo elemento
    current_row = [zipped[0]]
    
    # Usiamo l'altezza del primo elemento come riferimento per la tolleranza
    # Se un elemento è più in basso di "mezza cella", è una nuova riga
    first_h = zipped[0][1][3]
    y_threshold = first_h // 2 
    
    # 3. Loop intelligente
    for i in range(1, len(zipped)):
        current_cnt, current_box = zipped[i]
        prev_cnt, prev_box = zipped[i-1]
        
        current_y = current_box[1]
        prev_y = prev_box[1] # Y dell'elemento precedente nel sorting
        
        # LOGICA: Se la differenza di altezza tra questo e il precedente è piccola...
        if abs(current_y - prev_y) < y_threshold:
            # ... siamo ancora sulla stessa riga!
            current_row.append((current_cnt, current_box))
        else:
            # ... differenza troppo grande! È iniziata una NUOVA RIGA.
            
            # A. Chiudiamo la riga vecchia: la ordiniamo per X (Left -> Right)
            current_row.sort(key=lambda b: b[1][0])
            
            # B. Salviamo i contorni ordinati nella lista finale
            for item in current_row:
                final_contours.append(item[0])
            
            # C. Ricominciamo una nuova riga col contorno corrente
            current_row = [(current_cnt, current_box)]

    # 4. GESTIONE DELL'ULTIMA RIGA (che rimane fuori dal ciclo)
    current_row.sort(key=lambda b: b[1][0])
    for item in current_row:
        final_contours.append(item[0])

    return final_contours