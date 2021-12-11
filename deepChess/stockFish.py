# +
#
# stockFish.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class for interaction with stockFish
#
# -

import random
from stockfish import Stockfish

default_parameters = {
    "Write Debug Log": "false",
    "Skill Level": 10,
    "Minimum Thinking Time": 10
}


class stockFish_connector():
    '''
        stockFish connector
        Class for interaction with stockFish
    '''
    
    def __init__ (self, stockFish_path, parameters = {}):
        '''
            Arguments :
                - Path of the stockFish
                - Dictionnary of parameters
        '''
        
        # Loading stockFish
        
        ## Updating parameters with user preferences
        self.sf_parameters = default_parameters
        for key, value in parameters.items():
            self.sf_parameters[key] = value
            
        ## Loading stockFish
        self.sf = Stockfish(stockFish_path, parameters = self.sf_parameters)
        
    def play_move (self, move, top_k = 5):
        
        '''
            Play a move to stockFish
            The move should be in the format [a-h][1-8][a-h][1-8] with the source and destination
            Then, stockFish will play a move between the top_k with equal probability.
            
            Will be returned a tuple containing :
                - The played moved
                - If there is a mate
            Return a None if the move is not correct
                
            Arguments :
                - move (str) : coordonate of the move
                - top_k (int) : number of move to consider
        '''
        
        # Check if the move is correct before playing it
        if (self.sf.is_move_correct()):
        
            # Playing the move
            self.sf.make_moves_from_current_position([move])

            # Getting the next move
            moves = self.sf.get_top_moves(top_k)

            # Getting the state and the move
            n_moves = len(moves)
            move_id = random.randint(0,n_moves-1)

            (new_move, is_mate) = (moves[move_id]["Move"], moves[move_id]["Mate"])

            # Playing the move
            self.sf.make_moves_from_current_position([new_move])

            return new_move, is_mate
        else:
            # Incorrect move
            return None
        
    def get_board (self):
        
        return self.sf.get_board_visual()
