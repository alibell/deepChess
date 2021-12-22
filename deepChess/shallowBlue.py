# -*- coding: utf-8 -*-
# +
#
# shallowBlue.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Interface for communication with the shallow blue software
# Shallow blue is an open-source UCI chess engine written in C++ by : 
# Shallow blue : https://github.com/GunshipPenguin/shallow-blue written by Rhys Rustad-Elliott
#
# -

import subprocess
import select
import numpy as np


class shallowBlue_connector():
    
    """
        shallowBlue connector
        Class for interaction with shallowBlue    
    """
    
    def __init__ (self, shallowBlue_path, first_move = 0):
        
        '''
            Arguments :
                - Path of shallowblue
                - first_move : who played the first move, for move conversions
        '''
        
        # Loading shallowBlue
        self.sb = subprocess.Popen(shallowBlue_path, encoding='utf-8', stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        self.sb.stdout.readline() # Releasing the content

        
    def get_fen_position(self):
        
        """
            Get the fen position of the shallow blue board
        """
        
        res = self._get_fen_position_line()
        
        # Getting visual board
        board = [x[4:-2] for x in res[1:-2]]
        
        for i in [0,1,2,3]:
            board[i] = board[i].split("       ")[0]

        visual_board = np.array([list(x.replace(" ","").replace(".","0")) for x in board])
        visual_board[visual_board == ["0"]] = "" # The empty space will be empty
            
        # Creating the fen string

        ## The players position
        fen_list = []
        n_empty = 0

        for line in visual_board:
            line_fen = "" 
            for value in line:
                if value == '':
                    n_empty += 1 # Updating n_empty
                else:
                    # Releasing n_empty
                    if n_empty > 0:
                        line_fen += str(n_empty)
                        n_empty = 0
                    line_fen += value
            # Releasing n_empty
            if n_empty > 0:
                line_fen += str(n_empty)
                n_empty = 0
            fen_list.append(line_fen)
            
        ## The color
        if res[1].split("     ")[1].strip().split(" ")[0] == "White":
            color = "w"
        else:
            color = "b"
            
        ## Castle state
        castles = res[3].split("     ")[1].strip().split(" ")[-1].split("\n")[0]
        
        ## En passant
        en_passant = res[4].split(" ")[-1].split("\n")[0]

        ## Passant move
        no_progress = res[2].split(" ")[-1].split("\n")[0]

        ## Turn
        turn = int(no_progress)*2
        
        fen_string = '/'.join(fen_list)
        fen_string += f' {color}'
        fen_string += f' {castles[first_move]+castles[1-first_move].lower()}'
        fen_string += f' {en_passant}'
        fen_string += f' {no_progress} {turn}'

        return fen_string
    
    def get_next_moves (self):
        
        """
            Print the moves according to the current board 
        """
        
        moves = self._get_next_moves_line()
        
        return moves
    
    def _get_next_moves_line (self):
        
        """
            Get next moves line from the software
        """
        
        self.sb.stdout.flush()
        self.sb.stdin.flush()
        self.sb.stdin.write("printmoves\n")
        self.sb.stdin.flush()

        moves = self.sb.stdout.readline()
        moves = [x for x in moves.split("\n")[0].split(" ") if x != ""]
        
        return moves
    
    def get_moves_from_fen_position (self, fen_string):
        
        """
            Get a move according to a fen position
            
            Input : fen position string
        """
        
        self.set_fen_position(fen_string)
        moves = self.get_next_moves()
        
        return moves
    
    def set_fen_position (self, fen_string):
        
        """
            Set the current board to a fen position
            
            Input : fen position string
        """
        
        self.sb.stdin.flush()
        self.sb.stdin.write(f"position fen {fen_string}\n")
        self.sb.stdin.flush()

    def is_mate (self, fen_string = None):

        """
            Get if the current board is in mate
            If fen_string is None : the current board is evaluated without any change
        """

        if fen_string is not None:
            self.set_fen_position(fen_string)

        self.sb.stdout.flush()
        self.sb.stdin.flush()
        self.sb.stdin.write("go depth 1\n")
        self.sb.stdin.flush()
        state = self.sb.stdout.readline()
        self.sb.stdout.readline() # Releasing last line

        mate = state.split("mate")
        if len(mate) > 1:
            is_mate = (mate[1].strip().split(" ")[0])[0] == '-'
        else:
            is_mate = False

        return is_mate

    def quit (self):

        self.sb.stdout.flush()
        self.sb.stdin.flush()
        self.sb.stdin.write("quit\n")

        self.sb.kill()        
    
    def _get_fen_position_line(self):
        
        """
            Get fen position lines from the software
        """
        
        self.sb.stdout.flush()
        self.sb.stdin.flush()
        self.sb.stdin.write("printboard\n")
        self.sb.stdin.flush()
        
        res = []
        while True:
            line = self.sb.stdout.readline()
            res.append(line)

            if len(res[-1]) > 3:
                if res[-1][-2] == 'H':
                    break 
                
        return res

    def __deepcopy__ (self, copy):
        return self
