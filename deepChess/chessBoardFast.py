# -*- coding: utf-8 -*-
# +
#
# chessBoardFast.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Modelisation the chess board and chess rules
# This version is the same as the previous one but call the shallowBlue for move calculation
#
# -

from .errors import raiseException
from .shallowBlue import shallowBlue_connector
from .chessBoard import playChess as playChessSlow
import numpy as np_cpu # Seperating numpy and cupy because cupy doesn't support text
import numpy as np # Cupy support is still experimental, stay on numpy
import copy

##
# Main class
##

class playChess (playChessSlow):
    
    """
        Each instance of this class is a party of chess
        The class contains :
            A matrix representation of the chess board
            A method to make a move
            A method to evaluate if the game is won
        The chess board is representated 
    """
    
    def __init__ (self, sb_path, load_state = None):

        """
            sb_path : path of the shallowBlue binary
            load_state :
                None if the chess is initialized as a new game
                (board, turn) : if the chess is initialized from a current position, then board is the board matrix and turn a binary integer representing the next player
        """

        # Loading sb_path
        self._sb_path = sb_path
        self.sb = shallowBlue_connector(sb_path)

        # Initial fen
        self.fen = None
        
        # Loading the parent 
        super(playChess, self).__init__(load_state)

    
    def _getNextMove (self, board, player, ignore_check = False, opponentNextMoves = None, previousMove = None, depth = 1, alwaysDiag = False):
        """
            For a given board, given turn and given player
            Return the ensemble of possible moves
            
            Input :
                board : board in matrix representation
                player : actual player
                ignore_check : compute position ignoring the check
                opponentNextMoves : list of opponentNextMoves [useful to compute it only once when verifying if in check]
                previousMove : array of the previous move, previous move needed to compute the "prise au passant", by default the game previous move is used
                depth : int, depth of the function call, used to stop the loop with is_check call
                alwaysDiag : by default False, if true, the pion diagonal is always possible (use for fast checking of king check). By extension, is disable the remove collision filter.
                
            Output :
                move_layers : a list of possible move for each layer
                    Each move is represented by a list of two tuple : previous and new position
        """
        
        # Get fen position
        fen = self.get_fen_position()
        
        # Get next moves
        moves_sf = self.sb.get_moves_from_fen_position(fen)

        # Convert next moves
        moves = dict([(x, [self._stockFishToMove(x, 1-self.first_player)]) for x in moves_sf])
        if len(moves) == 0:
            moves = {0:[], 1:[]}

        return moves

    def _getOpponentsCurrentNextMove (self):

        """
            Disabling this function
        """

        return None

    def _is_check(self, board, player, opponentNextMoves = None, previousMove = None, depth = 1):
        
        """
            Compute for a board configuration if a player is in a check situation
            
            Input :
                board : matrix of the current board
                player : id of the player
                opponentNextMoves : list of opponentNextMoves [useful to compute it only once], if not given it is computed
                previousMove : array of the previous move, used to determine the "prise au passant", if None the current game previousMove is used
                depth : depth of the function call, used to stop the loop with getNextMove
                
            Return : boolean of check situation
        """
        
        ## Instead we just look if mate

        fen = self.get_fen_position()
        is_mate = self.sb.is_mate(fen)
        is_check = is_mate

        return is_check

    def is_mate (self):

        """
            Check if the current board is in mate
        """

        is_mate = self.is_check()

        return is_mate

    def copy (self):

        # Create a copy of the current chess game

        new_chess = copy.deepcopy(self)

        return new_chess