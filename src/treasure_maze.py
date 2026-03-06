from aima.search import Problem
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
