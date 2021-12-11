# -*- coding: utf-8 -*-
# +
#
# chessBoard.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Modelisation the chess board and chess rules
#
# -

from errors import raiseException
import numpy as np
import random
from functools import reduce
from operator import add


# +
# Convention :
## 0 : Is white, 1 is black
## 1 : Pawn, 2 : Knight, 3 : Bishop, 4 : Rook, 5 : Queen, 6 : King
## The player is always white and always in the bottom of the matrix
## Example : 21 is a black knight, 0 is the absence of piece
### Thus, we store a full chess board with a 8x8 matrix containing the 64 8-bit integer : 512 bit
# -

class playChess ():
    
    """
        Each instance of this class is a party of chess
        The class contains :
            A matrix representation of the chess board
            A method to make a move
            A method to evaluate if the game is won
        The chess board is representated 
    """
    
    # Knownleage about chess
    
    cav_move = np.array([
        [2,1], [2,-1],
        [-1,2],[1,2],
        [-1,-2],[1,-2],
        [-2,-1],[-2,1]
    ])
    cav_move = cav_move.reshape(8,2,1)
    bishop_move = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
    rook_move = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    king_move = np.concatenate([bishop_move, rook_move])
    
    def __init__ (self, load_state = None):
        """
            load_state :
                None if the chess is initialized as a new game
                (board, turn) : if the chess is initialized from a current position, then board is the board matrix and turn a binary integer representing the next player
        """
        
        # Pre-computing board characteristics
        
        self._initial_board = np.zeros((8,8), dtype = np.int8)
        self._initial_board[0,:] = [41, 21, 31, 51, 61, 31, 21, 41]
        self._initial_board[1,:] = 11
        self._initial_board[-2,:] = 10
        self._initial_board[-1,:] = self._initial_board[0,:]-1
        
        self._inital_board_pieces = self.__count_piece(self._initial_board)     
        
        # id of player layers in the layers board
        self.__player_layers_positions = dict([
            (0,[x for x in range (len(self._inital_board_pieces.keys())) if x%2 == 0]),
            (1,[x for x in range (len(self._inital_board_pieces.keys())) if x%2 != 0])
        ])
        
        # Board history : Keep two board history, the unique matrix and the multiple layers ones
        ## The multiple layers one is composed of one layer per piece and player
        self.board_history = []
        self.board_layers_history = []        
        
        # Initialization of the chess board
        if (load_state is not None):
            if self._checkChessBoard(load_state[0]) == False:
                raiseException("invalid_board")
            if load_state[1] not in [0,1]:
                raiseException("invalid_board")

            self.board, self.current_player = load_state
        else:
            self.board = self._initial_board
            self.current_player = random.randint(0,1)
            
        ## Sending it to board history
        self._memorize_board(self.board)
        
        # Initialization of number of turn
        self.turn = 1
    
    ###
    ### Ensemble of functions for analyzing the chess board
    ###
    
    ### Board composition
    
    def __count_piece (self, board):
        
        """
            __count_piece :
                Count the number of each piece inside the chess board
                
                Input : matrix of chess board
                Output : dictionnary containing in key the piece and in value the number of piece
        """
        
        pieces = dict([(x, (board == x).sum()) for x in np.unique(board) if x != 0])
        
        return(pieces)
        
    
    def _checkChessBoard(self, board):
        
        """
            _checkChessBoard :
                Check the validity of a chess board
                To be valid :
                    - It should be a matrix of 8x8
                    - Composed of 0, 10, 20, 30, 40, 50, 60, 11, 21, 31, 41, 51, 61
                    - With a respective limit of 8, 2, 2, 2, 1, 1 for each
                    
                Input : matrix of chess board
                Output : True if valid, false otherwise
        """
        
        valid = True
        
        # Check the size of the board
        if board.shape != self._initial_board:
            valid = False
        
        ## @TODO : fix this to consider promotion
        
        # Check the number of element
        pieces = self.__count_piece(board)
        if ((np.array(list(_inital_board_pieces.values()))-np.array(list(pieces.values()))) < 0).sum() > 0:
            valid = False
        
        return valid
    
    ### Moves
    
    def getNextMove (self):
        """
            For the actual board, return the possible next moves
            Output : list composed of next moves.
                     Each move is composed of a list of source coordonates and target coordonates
        """
        
        moves = self._getNextMove(self.board_layers_history[-1], self.current_player, self.turn)
        moves_list = reduce(add, list(moves.values()))
        
        return moves_list
    
    def _getNextMove (self, board_layers, player, turn):
        """
            For a given board, given turn and given player
            Return the ensemble of possible moves
            
            Input :
                board_layers : board in layer representation
                player : actual player
                turn : actual turn
            Output :
                move_layers : a list of possible move for each layer
                    Each move is represented by a list of two tuple : previous and new position
        """
        
        current_player_positions = self._playChess__player_layers_positions[player]
        current_opponent_positions = self._playChess__player_layers_positions[1-player]

        # Get the positions
        positions = np.array([np.where(x == 1) for x in board_layers], dtype = "object")
        player_position = [np.stack(x) for x in positions[current_player_positions]]
        opponent_position = [np.stack(x) for x in positions[current_opponent_positions]]

        moves = {}
        move_functions = {
            0:self._getPionMove,
            1:self._getCavMoves,
            2:self._getBishopMoves,
            3:self._getRookMoves,
            4:self._getLadyMoves,
            5:self._getKingMoves
        }

        for i in range(6):
            if move_functions[i] is not None:
                moves[i] = move_functions[i](player_position, opponent_position, player, turn)
                moves[i] = moves[i].T.reshape(moves[i].shape[1], 2, 2).tolist()
            
        return moves
    
    def __removeOutOfBorderMoves (self, move_np):
        '''
            For internal purpuse
            Remove out of border moves
            In input : np array
            In output : np array
        '''
    
        moves = move_np[:,(move_np[2] <= 7) & (move_np[2] >= 0) & (move_np[3] <= 7) & (move_np[3] >= 0)]
    
        return moves
    
    def __removeCollisionMoves (self, move_np, player_position):
        '''
            For internal purpuse
            Remove out the collision move : moves in the position of current pieces
            In input : np array, player_position
            In output : np array
        '''
    
        player_position_concat = np.concatenate(player_position, axis = 1)
        moves_positions = move_np[[2,3],:]
        overlapping_player = np.where((moves_positions[:,None].T == player_position_concat.T).all(axis=2))[0]
        moves = np.delete(move_np, overlapping_player, axis = 1)
    
        return moves
    
    def _getPionMove (self, player_position, opponent_position, player, turn):
        
        """
            Compute the pion move according to :
                Current player position
                Opponent position
                Current player
                Current turn
                
            Output : list of moves
        """            
        current_range = 1 + int(turn == 1) # Possible range of the move
        direction = -(1-player)+(player)*1 # Direction of the move
        
        # Current position
        wp = player_position[0] # Get the player position
        wp_repeat = np.concatenate([
            wp for j in range(current_range)
        ], axis = 1) # Repeat it to cover the possible positions
        
        # 1. Computing classical move : straight forward
        
        npos = np.stack([
            np.concatenate([wp[0,:]+direction*(j+1) for j in range(current_range)]).flatten(),
            wp_repeat[1,:]
        ])
        
        moves = np.concatenate([
            wp_repeat,
            npos
        ])
        
        # 2. Computing diagonal moves
        all_oponents = np.concatenate(opponent_position, axis = 1)
        all_diagonal = np.stack([
            np.concatenate([wp[0,:]+direction*(1) for j in list([-1,1])]).flatten(),
            np.concatenate([wp[1,:]+j*(1) for j in list([-1,1])]).flatten()
        ])
        
        possible_moves = np.concatenate([
                wp_repeat,
                all_diagonal
        ])[:,np.where((all_oponents[:,None].T == all_diagonal.T).all(axis=2))[0]]
        
        if possible_moves.shape[-1] != 0:
            moves = np.concatenate([
                moves,
                possible_moves
            ], axis = 1)
        
        # Removing out of board moves
        moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
        
        return moves
    
    def _getCavMoves (self,  player_position, opponent_position, player, turn):
        
        # Current position
        wp = player_position[1] # Get the player position
        wp_repeat = np.concatenate([
            wp for j in range(8)
        ], axis = 1) # Repeat it to cover the possible positions
        
        # Getting new positions
        npos = np.concatenate([wp + cav_move[j] for j in range(8)], axis = 1)
        
        # Concatenating with wp
        moves = np.concatenate([
            wp_repeat,
            npos
        ])
        
        moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
        
        return moves
    
    def _getLongRangeMoves (self, rule, wp, player_position, opponent_position, player, turn):
        """
            Genering function for long range moves
        """

        # Current position
        moves = []
        
        player_position_concat = np.concatenate(player_position, axis = 1)
        opponent_position_concat  = np.concatenate(opponent_position, axis = 1)

        # Looping over all possible positions
        for pos in wp.T:

            # Set True for all positions
            directions = np.array([True for i in range(len(rule))])
            new_pos = pos
            pos = pos.reshape(2,1)
            n_move = 1

            while directions.sum() != 0:

                # Apply a movement for each direction
                new_pos = new_pos+rule[directions]

                # Getting overlapping positions
                overlapping_player = np.where((new_pos[:,None] == player_position_concat.T).all(axis=2))[0]
                overlapping_opponent = np.where((new_pos[:,None] == opponent_position_concat.T).all(axis=2))[0]

                # We disable the overlapping positions
                directions[np.concatenate([overlapping_opponent, overlapping_player])] = False

                # For overlapping player, we disable the move
                if (len(overlapping_player) > 0):
                    new_pos = np.delete(new_pos, overlapping_player, axis = 0)

                # Storing the positions
                new_move = np.concatenate([
                    np.concatenate([pos for n in range(len(new_pos))], axis = 1),
                    new_pos.T
                ])
                moves.append(new_move)

                # For overlapping opponent, we disable the move for next loop
                if (len(overlapping_opponent) > 0):
                    new_pos = np.delete(new_pos, overlapping_opponent, axis = 0)

                # We stop after 8 moves
                if n_move == 8:
                    break

                # Go for net loop
                n_move += 1
        
        moves = np.concatenate(moves, axis = 1)
        moves = self.__removeOutOfBorderMoves(moves)
        
        return moves
        
    def _getBishopMoves (self, player_position, opponent_position, player, turn):
        
        # Current position
        wp = player_position[2] # Get the player position
        moves = self._getLongRangeMoves(bishop_move, wp ,player_position, opponent_position, player, turn)
        
        return moves
 
    def _getRookMoves (self,  player_position, opponent_position, player, turn):
        
        # Current position
        wp = player_position[3] # Get the player position
        moves = self._getLongRangeMoves(rook_move, wp ,player_position, opponent_position, player, turn)

        return moves
    
    def _getLadyMoves (self, player_position, opponent_position, player, turn):
        
        # Mix of Bishop and Rook, easy one
        wp = player_position[4] # Get the player position
        bishop_moves = self._getLongRangeMoves(bishop_move, wp ,player_position, opponent_position, player, turn)
        rook_moves = self._getLongRangeMoves(rook_move, wp ,player_position, opponent_position, player, turn)
        
        moves = bishop_moves + rook_moves
        
        return moves
    
    def _getKingMoves (self, player_position, opponent_position, player, turn):
        
        wp = player_position[5] # Get the player position
        
        moves = np.concatenate([
            np.concatenate([wp for i in range(8)], axis = 1),
            king_move.T+wp
        ])
        
        moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
        
        return moves
    
    ###
    # Ensemble of functions for transforming the representation of the board
    ###
    
    def _board_to_layer (self, board):
        """
            _board_to_layer :
                Convert the current board to a layer representation of it
                Each layer represent the position of each piece for each player such as it is done in the alpha zero project
            
                - Input : board matrix
                - Ouput : 3D numpy array with each piece and player position
                
                @TODO : Maybe try to do it with scipy.parse
        """
        
        layer = np.array([(board == x).astype("int8") for x in list(self._inital_board_pieces.keys())])
        
        return layer
    
    def _memorize_board (self, board):
        
        """
            _memorize_board :
                Keep board state in memory
                For each board state, it compute the layer representation and both of them are stored in its history
                
                Input : board to store
                Output : None
        """
        
        # Keep a board state in the history
        board_layers = self._board_to_layer(board)
        
        # Storing boards
        self.board_history.append(board)
        self.board_layers_history.append(board_layers)
        
        pass
    
    ###
    # Ensemble of methods to play
    ###
    
    def play_move(self, move):
        
        """
            play_move :
                Play a move
                For that :
                    1. Check the validity of the move
                    2. Play the move
                    3. Analyse the consequence of the move
                    4. Storing and recomputing the new state
                    5. Update the player and eventually the turn
                    6. Promote pieces
                    7. Evaluate if there is a winner
        """
        
        # Getting possible moves
        moves = self.getNextMove()
        
        # Check the move
        if move not in moves:
            raiseException("incorrect_move")
        
        # Play the move
        new_board = self.board.copy()
        
        piece = new_board[tuple(move[0])] 
        new_board[tuple(move[0])] = 0 # Setting empty the previous position
        new_board[tuple(move[1])] = piece # Setting the new position
        
        # Storing and recomputing the new state
        # self._memorize_board(new_board)
        
        return new_board

chess = playChess()
next_move = chess.getNextMove()

chess.play_move(next_move[0])
