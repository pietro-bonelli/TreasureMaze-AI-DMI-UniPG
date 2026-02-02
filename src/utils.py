# Funzione per plot corretto delle immagini
import cv2
import numpy as np
from matplotlib import pyplot as plt

def plotImage(image, title, saveToFile = False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # per evitare colori sballati
    #plt.imshow(image)
    #plt.title(title)
    #plt.axis('off')
    if saveToFile:
        plt.savefig('plots/{title}.png')
   # plt.show()

def show_pipeline(steps):
    """
    Mostra più immagini fianco a fianco in un'unica finestra.
    
    Parametri:
    steps -- Una lista di tuple: [(immagine, "Titolo"), (immagine2, "Titolo2"), ...]
    """
    num_images = len(steps)
    
    # Crea una "figura" con '1' riga e 'num_images' colonne
    # figsize=(width, height) imposta la dimensione della finestra in pollici
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    # Se c'è una sola immagine, 'axes' non è una lista, quindi lo rendiamo tale per uniformità
    if num_images == 1:
        axes = [axes]
        
    for i, (image, title) in enumerate(steps):
        ax = axes[i]
        
        # Gestione Colore vs Bianco e Nero
        if len(image.shape) == 2:
            # Immagine in scala di grigi (2 dimensioni: H, W)
            ax.imshow(image, cmap='gray')
        else:
            # Immagine a colori (3 dimensioni: H, W, Canali)
            # Converti da BGR (OpenCV) a RGB (Matplotlib)
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_img)
            
        ax.set_title(title)
        ax.axis('off') # Rimuove i righelli
        
    plt.tight_layout() # Ottimizza gli spazi bianchi tra le immagini
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