import cv2
import numpy as np
import os

# S = Start, T = Treasure, X = Muro, 1/2/3/4 = Costi
maze_layout = [
    ['S', '1', 'X', '1'],
    ['1', 'X', '2', '1'],
    ['1', '1', 'T', 'X'],
    ['X', '1', '1', '4']
]

cell_size = 100
rows = len(maze_layout)
cols = len(maze_layout[0])
img_size = (cols * cell_size, rows * cell_size)

# Crea immagine bianca
image = np.ones((img_size[1], img_size[0]), dtype=np.uint8) * 255

# Disegna griglia e caratteri
for r in range(rows):
    for c in range(cols):
        # Coordinate angolo top-left della cella
        x0, y0 = c * cell_size, r * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        
        # Disegna il rettangolo (la cella)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0), 2) # 0 = nero, spessore 2
        
        # Scrivi il carattere al centro
        text = maze_layout[r][c]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        
        # Calcola dimensione testo per centrarlo
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x0 + (cell_size - text_w) // 2
        text_y = y0 + (cell_size + text_h) // 2
        
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0), thickness)

# Salva nella cartella assets
output_path = 'assets/test_maze.png'
cv2.imwrite(output_path, image)
print(f"Immagine di test creata: {output_path}")