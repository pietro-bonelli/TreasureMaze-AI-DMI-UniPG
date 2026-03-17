import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import plotImage, show_pipeline, orderPoints, sort_contours
from classifier import predict

image_path = "assets/test_maze.png"
image_path = "assets/test2_2.jpg"


def getBinaryImage(image):
    plotImage(image, "Immagine originale")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converto l'immagine in SCALA DI GRIGI, essenziale per analisi successive
    plotImage(gray_image, "Scala di grigi")

    ret, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV) # applico un threshold statico del "50%"
    plotImage(thresh1, "Immagine con Threshold classico")
    
    # Per applicare adaptive Threshold, occorre fare "blurring", ossia rimuovere il rumore dall'immagine.
    #blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    blur = cv2.medianBlur(gray_image, 5)
    plotImage(blur, "Immagine blurrata")

    thresh2a = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=7, C=2) # applico un threshold adattivo con tecnica "MEAN". blockSize = quanti pixel guardare intorno per deciere se un pixel è bianco o nero.
    plotImage(thresh2a, "Immagine con Threshold adattivo MEAN")
    thresh2b = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=3) # applico un threshold adattivo con tecnica "GAUS"
    plotImage(thresh2b, "Immagine con Threshold adattivo GAUSSIAN")

    # Cerco di "chiudere" i buchi nei perimetri dei box
    kernel_small = np.ones((3,3), np.uint8)
    kernel_large = np.ones((5,5), np.uint8)
    thresh2b_morph = cv2.dilate(thresh2b, kernel_small, iterations=1)
    thresh2b_morph = cv2.morphologyEx(thresh2b_morph, cv2.MORPH_CLOSE, kernel_large)
    plotImage(thresh2b_morph, "Immagine con Threshold + Morph")

    # Disegno un super rettangolo sul perimetro esterno dell'immagine, così da non perdere il perimetro se l'immagine dovesse essere tagliata
    h, w = thresh2b_morph.shape[:2] # w,h sono le coordinate dell'angolo in basso a destra dell'immagine
    cv2.rectangle(thresh2b_morph, (0, 0), (w, h), 255, thickness=5) # diesgna contorno dai vertici 0,0 e 2,h, colore bianco, spessore 5px

    '''show_pipeline([
        (image, 'Immagine originale'),
        (gray_image, 'Scala di grigi'),
        (thresh1, 'Threshold statico'),
        (blur, 'Blur'),
        (thresh2a, 'Threshold MEAN'),
        (thresh2b, 'Threshold GAUS'),
        (thresh2b_morph, 'Morph')
    ])'''

    return thresh2b_morph, gray_image

def findBoxes(binaryImage, originalImage, grayImage):
    # Calcolo le aree
    img_h, img_w = binaryImage.shape[:2]
    total_area = img_h * img_w

    # Approssimativamente ogni cella sarà dal 5% al 15% dell'area totale dell'immagine.
    min_area = total_area * 0.005
    max_area = total_area * 0.15

    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # RETR_TREE crea una gerarchia (quadrati dentro altri quadrati sono considerati "figli")
    # CHAIN_APPROX_SIMPLE per ottimizzazione RAM: non salva in memoria tutti i punti di una linea ma solo i 2 estremi.
    contoursImg = originalImage.copy()
    
    validContours = [] 
    croppedImages = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area < min_area or area > max_area):
            continue
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.04 * perimeter # margine di errore tollerato.
        approx = cv2.approxPolyDP(cnt, epsilon, True) # riduce il numero di vertici di una linea (approssimandola)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Controllo l'Aspect Ratio della forma (lati devono essere simili)
            x,y,w,h = cv2.boundingRect(approx)
            ratio = float(w) / float(h)
            if 0.7 <= ratio <= 1.3: # tolleranza del 30%
                validContours.append(approx)

    # Ordinamento delle celle
    if len(validContours) > 0:
        validContours = sort_contours(validContours) 
    else:
        print("Nessuna cella trovata")
        return []

    for i, box in enumerate(validContours):
        # Disegno il contorno (Verde)
        cv2.drawContours(contoursImg, [box], -1, (0, 255, 0), 3)
    
        rect = orderPoints(box)
        dst_pts = np.array([
            [0, 0],
            [28, 0], 
            [28, 28], 
            [0, 28] 
        ], dtype="float32")

        # Sistemo la prospettiva della foto per "appiattirla"
        M = cv2.getPerspectiveTransform(rect, dst_pts) # calcola una matrice di trasformazione
        warped = cv2.warpPerspective(grayImage, M, (28, 28), flags=cv2.INTER_AREA) # Utilizza la matrice per appiattire l'immagine e trasformarla in un quadrato quasi "perfetto"
        ret, warped_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = np.ones((2,2), np.uint8)
        warped_bin = cv2.dilate(warped_bin, kernel, iterations=1)
        #warped_bin = cv2.GaussianBlur(warped_bin, (1, 1), 0)
        croppedImages.append(warped_bin)
    
    # Visualizzazione
    debug_steps = [(contoursImg, "Celle individuate")]
    plotImage(contoursImg, "Contorni")
    for i in range(len(croppedImages)):
        debug_steps.append((croppedImages[i], f"Cella {i}"))
        
    show_pipeline(debug_steps)
    return croppedImages


image = cv2.imread(image_path) # carico l'immagine dal file
binaryImage, grayImage = getBinaryImage(image)
boxes = findBoxes(binaryImage, image, grayImage)

labirinto = []
debug_steps = []
for box in boxes:
    prediction = predict(box)
    labirinto.append(prediction)
    print(prediction)
    debug_steps.append((box, f"Previsione: {prediction}"))
show_pipeline(debug_steps)
