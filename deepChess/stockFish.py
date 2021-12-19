# +
#
# stockFish.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class for interaction with stockFish
#
# -

import random
import string
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
        
    def get_top_moves (self, top_k = 5):
        
        """
            A simple wrapper for SF get_top_moves functions
            
            Input :
                Number of moves to get
            Ouput :
                Moves
        """
        
        moves = self.sf.get_top_moves(top_k)
        moves = [dict(zip(
                ["move","promotion","information"],
                [x["Move"][0:4].lower(), x["Move"][-1].lower() if x["Move"][-1].lower() in ["r","n","b","q"] else None, x["Mate"]]
            ))
            for x in moves]
        
        return moves
        
    def _play_move (self, top_k = 5):
        '''
            Playing a move without any sanity check.
            Used for internal use purpose.

            Will be returned a tuple containing :
                - The played moved
                - If there is a mate            

            Input : top_k : number of move to consider
        '''
        
        # Getting the next move
        moves = self.get_top_moves(top_k)

        # Getting the state and the move
        n_moves = len(moves)
        move_id = random.randint(0,n_moves-1)
        
        new_move = moves[move_id]

        # Playing the move
        self.sf.make_moves_from_current_position([new_move])        
        
        return next_move
        
    def play_move (self, top_k = 5):
        
        '''
            Let stockFish play a move first.
            
            Will be returned a tuple containing :
                - The played moved
                - If there is a mate

            Input : top_k : number of move to consider
            Return a None if the move is not correct
        '''
        
        next_move = self._play_move(top_k)
        
        return next_move
        
    def send_move (self, move, top_k = 5):
        
        '''
            Send a move to stockFish
            The move should be in the format [a-h][1-8][a-h][1-8] with the source and destination
            Then, stockFish will play a move between the top_k with equal probability.
            
            Will be returned a tuple containing :
                - The played moved
                - If there is a mate
            Return a None if the move is not correct
                
            Arguments :
                - move (str) : coordonate of the move
                - top_k (int) : number of move to consider
                
            Return a dictionary containing :
                - The next move
                - If there is a promotion, the promotion, None otherwise
                - Information
        '''
        
        # Check if the move is correct before playing it
        if (self.sf.is_move_correct(move)):
        
            # Playing the move
            self.sf.make_moves_from_current_position([move])

            next_move = self._play_move(top_k)

            return next_move
        else:
            # Incorrect move
            return None
        
    def get_board (self):
        
        return self.sf.get_board_visual()
