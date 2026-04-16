import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aima.search import Problem, astar_search
import math
from typing import NamedTuple, Tuple

Position = Tuple[int, int]
class State(NamedTuple):
    pos: Position
    treasures: Tuple[Position, ...]
    walls: Tuple[Position, ...]

class TreasureMaze(Problem):
    def __init__(self, predictions_list: list, k=None) -> None:
        """
        k = numero di tesori da trovare. k = None indica trovare tutti i tesori.
        """
        self.total_cells = len(predictions_list)
        self.size = int(math.sqrt(self.total_cells)) # Larghezza e Altezza labirinto
        self.grid = []
        self.treasures = set()

        if self.size * self.size != self.total_cells:
            raise ValueError(f"Errore: {self.total_cells} non permettono la formazione di una griglia quadrata.")
    
        start_pos = None
        for i in range(self.size):
            row = []
            for j in range(self.size):
                val = predictions_list[(i*self.size) + j] # elemento per la riga j colonna i.
                row.append(val)
                
                if val == 'S':
                    if not start_pos:
                        start_pos = (i, j)
                    else:
                        raise Exception("Errore: Individuati più punti di START nel labirinto.")
                elif val == 'T':
                    self.treasures.add((i, j)) # aggiunto il tesoro alla lista di tesori
            self.grid.append(row)
        if not start_pos:
            raise ValueError("Non è presenta nessun punto di START nella griglia.")
        
        if k is None:
            self.k = len(self.treasures) # Se k = None (trova tutti i tesori) -> k diventa numero totale di tesori trovati.

        initial_state = State(pos=start_pos, treasures=(), walls=())
        super().__init__(initial_state)

    # Dato uno stato (i, j) ritorna la lista delle possibili azioni.
    def actions(self, state: State) -> list:
        """
        Lista delle azioni possibili a partire dallo stato 'state'.
        La lista delle azioni è ritornata sottoforma di array di coordinate che rappresentano lo spostamento.
        """
        (i, j) = state.pos
        #                      SU     GIU    SINISTRA  DESTRA
        possible_actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if i == 0: # mi trovo sulla prima riga del labirinto
            possible_actions.remove((-1, 0))
        if i == self.size - 1: # mi trovo sull'ultima riga del labirinto
            possible_actions.remove((1, 0))
        if j == 0: # mi trovo sulla prima colonna del labirinto
            possible_actions.remove((0, -1))
        if j == self.size - 1: # mi trovo sull'ultima colonna del labirinto
            possible_actions.remove((0, 1))
        return possible_actions

    def goal_test(self, state: State) -> bool:
        if len(state.treasures) >= self.k: # Goal Test passa se sono stati trovati almeno k tesori (dove k = N. Tesori sulla mappa se non definito)
            return True
        else:
            return False
    
    def result(self, state: State, action: tuple) -> State:
        """ Dato uno stato 'state' e un azione 'action' ritorna un nuovo stato risultante dall'azione."""
        (i, j) = state.pos
        ai, aj = action
        new_pos = (i+ai, j+aj)

        new_collected = list(state.treasures)
        new_walls = list(state.walls)
        if new_pos in self.treasures and new_pos not in new_collected: # controllo se mi trovo in un tesoro, e se eventualmente lo avessi già preso.
            new_collected.append(new_pos)
            new_collected.sort()
        elif self.grid[new_pos[0]][new_pos[1]] == 'X' and new_pos not in new_walls:
            new_walls.append(new_pos)
            new_walls.sort()
        
        return State(new_pos, tuple(new_collected), tuple(new_walls))
    
    def path_cost(self, c: int, state1: State, action: tuple, state2: State):
        """ Calcola il costo per passare da 'state1' a 'state2' effettuando l'azione 'action.
        'c' = costo accumulato fin'ora."""
        pos2 = state2.pos
        walls1 = state1.walls

        cell = self.grid[pos2[0]][pos2[1]]
        if cell == 'X': # La nuova posizione è un muro
            if pos2 not in walls1: # il muro non è ancora stato abbattuto.
                return c + 5 # costo 5 per abbatterlo
            else:
                return c + 1 # se già abbattuto
        elif cell in ['S', 'T']:
            return c + 1
        else: # cella calpestabile con costo 1/4
            return c + int(cell)
        
    def h(self, node) -> float: 
        """Distanza dal m-esimo tesoro più vicino"""
        state = node.state
        collected = state.treasures
        m_missing = self.k - len(collected) # m = tesori ancora da raccogliere
        
        if m_missing <= 0:
            return 0.0
        
        missing_treasures = self.treasures - set(collected)
        distances = []
        i, j = state.pos
        for ii, jj in missing_treasures:
            dist = abs(i - ii) + abs(j - jj)
            distances.append(dist)
        distances.sort() # Costo O(nlogn)
        return float(distances[m_missing - 1]) # m-esimo tesoro più vicino

    def h_alternative(self, node) -> float:
        """Distanza di Manhattan dal tesoro (non raccolto) più vicino."""
        state = node.state
        pos = state.pos
        collected = state.treasures
        missing_k = self.k - len(collected)
        if missing_k <= 0: # se ho già trovato tutti i tesori necessari, l'euristica deve valere 0.
            return 0.0
        
        missing_treasures = self.treasures - set(collected) # coordinate dei tesori ancora da raccogliere.
        min_dist = float('inf')

        i, j = pos
        for ii, jj in missing_treasures:
            dist = abs(i - ii) + abs(j - jj) # distanza di Manhattan dal tesori
            if dist < min_dist:
                min_dist = dist
        return float(min_dist)
        

            
if __name__ == "__main__":
    # 1. Creiamo un labirinto fittizio 5x5 (array piatto da 25 elementi)
    # Per farti capire visivamente la griglia:
    # S 1 X 1 T
    # 2 X 1 X 1
    # 1 1 1 1 1
    # X X X 2 X
    # T 1 1 1 1
    
    dummy_predictions = [
        'S', '1', 'X', '1', 'T',
        '2', 'X', '1', 'X', '1',
        '1', '1', 'T', '1', '1',
        'X', '1', 'X', '2', 'X',
        'T', '1', '1', '1', 'T'
    ]

    print("Inizializzazione del labirinto...")
    try:
        # Vogliamo trovare TUTTI i tesori, quindi k=None
        maze_problem = TreasureMaze(dummy_predictions, k=None)
        print(f"Labirinto {maze_problem.size}x{maze_problem.size} caricato.")
        print(f"Tesori da trovare: {maze_problem.k} in posizioni {maze_problem.treasures}")
        
        # 2. Eseguiamo l'algoritmo A*
        print("\nRicerca del percorso ottimale in corso (A*)...")
        goal_node = astar_search(maze_problem, h=maze_problem.h)
        
        # 3. Estrazione e stampa dei risultati
        if goal_node:
            print("\n--- SOLUZIONE TROVATA! ---")
            print(f"Costo totale del percorso: {goal_node.path_cost}")
            
            # Il metodo .path() di AIMA risale l'albero usando i puntatori 'parent'
            # e restituisce la lista dei nodi dall'inizio alla fine.
            percorso = goal_node.path()
            print(f"Numero totale di passi: {len(percorso) - 1}\n")
            
            print("Dettaglio mosse:")
            for step, node in enumerate(percorso):
                stato = node.state
                if step == 0:
                    print(f"[{step}] Partenza da {stato.pos}")
                else:
                    azione = node.action # Le coordinate dove ci siamo appena mossi
                    tesori_presi = len(stato.treasures)
                    muri_rotti = len(stato.walls)
                    costo_attuale = node.path_cost
                    
                    print(f"[{step}] Vai in {azione} | Costo accumulato: {costo_attuale} | Tesori in tasca: {tesori_presi} | Muri abbattuti: {muri_rotti}")
        else:
            print("\nNessuna soluzione trovata! Il labirinto è impossibile.")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")