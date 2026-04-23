import os
import time
from src.treasure_maze import TreasureMaze, show_path
from src.read_image import extract_treasure_maze_from_image
from aima.search import astar_search, breadth_first_graph_search

def run_pipeline():
    print("--- TreasureMaze ---")
    
    filename = input("Inserisci il nome dell'immagine (presente nella cartella 'assets'): ").strip()
    image_path = os.path.join("assets", filename)

    if not os.path.exists(image_path):
        print(f"Errore: Il file '{image_path}' non è stato trovato nella cartella 'assets'.")
        return

    try:
        print(f"\n[1/3] Analisi dell'immagine e classificazione in corso...")
        matrice_2d = extract_treasure_maze_from_image(image_path)
        
        # Trasforma matrice 2D in array 1D.
        predictions_flat = [cella for riga in matrice_2d for cella in riga]
        grid_size = len(matrice_2d)
        print(f"Successo: Rilevato labirinto {grid_size}x{grid_size}.")

        k_input = input("Quanti tesori raccogliere? (Premi invio per TUTTI): ").strip()
        k = int(k_input) if k_input.isdigit() else None
        
        maze_problem = TreasureMaze(predictions_flat, k=k)
        print(f"Gioco inizializzato. Obiettivo: raccogliere {maze_problem.k} tesori.")

        print(f"\n[2/3] Avvio dei solutori per il confronto...")
        risultati = {}


        # --- A* (Euristicha: Distanza dall'm-esimo tesoro non raccolto più vicino (avanzata)) ---
        maze_problem.expanded_nodes = 0
        t0 = time.time()
        goal_astar = astar_search(maze_problem, h=maze_problem.h)
        risultati['A*_avanzata'] = {
            'goal': goal_astar, 
            'time': time.time() - t0, 
            'nodes': maze_problem.expanded_nodes
        }

        print("\n" + "="*60)
        print(f"{'Algoritmo':<25} | {'Tempo (s)':<10} | {'Nodi Espansi':<12} | {'Costo'}")
        print("-" * 60)
        for algoritmo, data in risultati.items():
            res = data['goal']
            costo = res.path_cost if res else "N.D."
            print(f"{algoritmo:<25} | {data['time']:<10.4f} | {data['nodes']:<12} | {costo}")
        print("="*60)

        if risultati['A*_avanzata']['goal']:
            print(f"\n[3/3] Visualizzazione del percorso ottimale...")
            percorso = risultati['A*_avanzata']['goal'].path()
            show_path(maze_problem, percorso, f"Soluzione Ottima (Costo: {risultati['A*_avanzata']['goal'].path_cost})")
        else:
            print("\nNessuna soluzione trovata per questo labirinto.")

    except Exception as e:
        print(f"\n[ERRORE]: {e}")

if __name__ == "__main__":
    run_pipeline()