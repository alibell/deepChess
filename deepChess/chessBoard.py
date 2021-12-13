# -*- coding: utf-8 -*-
# +
#
# chessBoard.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Modelisation the chess board and chess rules
#
# -

from .errors import raiseException
import numpy as np_cpu # Seperating numpy and cupy because cupy doesn't support text
import numpy as np # Cupy support is still experimental, stay on numpy
import random
from functools import reduce
from operator import add
import string
import math


# +
# Convention :
## 0 : Is white, 1 is black
## 1 : Pawn, 2 : Knight, 3 : Bishop, 4 : Rook, 5 : Queen, 6 : King
## The player is always white and always in the bottom of the matrix
## Example : 21 is a black knight, 0 is the absence of piece
### Thus, we store a full chess board with a 8x8 matrix containing the 64 8-bit integer : 512 bit
# -

def np_delete (np_array, list_to_remove, axis = 0):
    
    """
        Faster implementation of np.remove, do it by getting the right index
    """
    
    if (len(list_to_remove) > 0):


        list_of_indexes = np.array(list(range(np_array.shape[axis])))
        mask = list_of_indexes[(np.isin(list_of_indexes, np.array(list_to_remove)) == False)]

        np_array = np.take(np_array, mask, axis = axis)
    
    return np_array


class playChess ():
    
    """
        Each instance of this class is a party of chess
        The class contains :
            A matrix representation of the chess board
            A method to make a move
            A method to evaluate if the game is won
        The chess board is representated 
    """
    
    def __init__ (self, load_state = None):
        """
            load_state :
                None if the chess is initialized as a new game
                (board, turn) : if the chess is initialized from a current position, then board is the board matrix and turn a binary integer representing the next player
        """
        
        # Knownledge about chess

        self.cav_move = np.array([
            [2,1], [2,-1],
            [-1,2],[1,2],
            [-1,-2],[1,-2],
            [-2,-1],[-2,1]
        ])
        self.cav_move = self.cav_move.reshape(8,2,1)
        self.bishop_move = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
        self.rook_move = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        self.king_move = np.concatenate([self.bishop_move, self.rook_move])
        self.board_dictionnary = np_cpu.array(["p","n","b","r","q","k"])
        self.board_coordonates = {
            "default":{
                "x": np_cpu.array([list(range(8))]).astype("U1"),
                "y": np_cpu.array([list(range(8))]).astype("U1").T
            }, "sf":{
                "x": np_cpu.array([list(string.ascii_lowercase[0:8])]).astype("U1"),
                "y": np_cpu.array([list(range(8, 0, -1))]).astype("U1").T
            }
        }
        
        # Knownledge about stockfish
        self._localToStockfish = tuple([
            dict(zip(list(range(8)), list(range(8,0,-1)))),
            dict(zip(list(range(8)), list(string.ascii_lowercase[0:8])))
        ]) # Useful for translation of stockfish move
        self._localToStockishMatrice = np_cpu.char.add(
            *np_cpu.meshgrid(
                np_cpu.array(list(self._localToStockfish[1].values())).astype('U1'),
                np_cpu.array(list(self._localToStockfish[0].values())).astype('U1')
            )
        ) # Pre-computation of translation matrice
        self._localToStockishMatrice_flip = np.flip(self._localToStockishMatrice, axis = 0).copy() # Flipped version of the translation matrice
        
        # Pre-computing board characteristics
        
        self._initial_board = np.zeros((8,8), dtype = np.int8)
        self._initial_board[0,:] = np.array([41, 21, 31, 51, 61, 31, 21, 41])
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
            self.first_player = random.randint(0,1)
            self.current_player = self.first_player
                        
        ## Initialization of number of turn
        self.turn = 1
        
        ## King and rook watcher
        ### Needed for the roque
        self._not_moved = {
            0:[[7, 7], [7, 0], [7,4]],
            1:[[0,0],[0,7],[0,4]]
        }
        
        self.has_roque = {
            0:False,
            1:False
        } # Store if the player has already play a roque
        
        ## State of the game
        self.mate = False # Play until is a mate
        self.winner = None

        ## Sending it to board history
        self._memorize_board(self.board)
        
        ## Pre-compute next moves
        self.nextMoves = self._getCurrentNextMove()
    
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
        
        pieces = dict([(x.tolist(), (board == x).sum().tolist()) for x in np.unique(board) if x != 0])
        
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
        if ((np.array(list(self._inital_board_pieces.values()))-np.array(list(pieces.values()))) < 0).sum() > 0:
            valid = False
        
        return True
    
    ### Moves
    
    def _getCurrentNextMove (self):
        
        """
            For the actual board, return the possible next moves
            Output : list composed of next moves.
                     Each move is composed of a list of source coordonates and target coordonates
        """
        
        moves = self._getNextMove(self.board, self.current_player)
        moves_list = reduce(add, list(moves.values()))
        
        return moves_list
    
    def _getOpponentsCurrentNextMove (self):
        
        """
            For the actual board, return the possible next moves of the opponents (1 move in the futur)
            Output : list composed of next moves.
                     Each move is composed of a list of source coordonates and target coordonates
        """
        
        moves = self._getNextMove(self.board, 1-self.current_player)
        moves_list = reduce(add, list(moves.values()))
        
        return moves_list
    
    def _movesToStockFish (self, moves, first_move):
        """
            Convert a list of local moves to a liste of stockFish move
            
            Input : list of moves, in the format : [[[y_source, x_source], [y_target, x_target]], ...]
                    first_move : needed to correctyle convert the coordonate, virtually flip the game if the opponent begins
            Output : list of stockFish moves
        """
        
        if first_move == False:
            _localToStockishMatrice = self._localToStockishMatrice_flip
        else:
            _localToStockishMatrice = self._localToStockishMatrice
        
        _move_coordonates = tuple(np.array(moves).T.tolist())
        moves = reduce(np_cpu.char.add, _localToStockishMatrice[_move_coordonates].tolist()).tolist()
        
        return moves
    
    def _stockFishToMove (self, stockfish, first_move):
        
        """
            Convert an unique stockFish move to a local one
            
            Input : stockFish move
                    first_move : needed to correctyle convert the coordonate, virtually flip the game if the opponent begins
            Output : local move
        """
        
        if first_move == False:
            _localToStockishMatrice = self._localToStockishMatrice_flip
        else:
            _localToStockishMatrice = self._localToStockishMatrice
            
        move = np_cpu.array([
            np_cpu.where(_localToStockishMatrice == stockfish[0:2]),
            np_cpu.where(_localToStockishMatrice == stockfish[2:4])
        ]).reshape(2,2).tolist()
        
        return move
    
    def _getNextMove (self, board, player, ignore_check = False, opponentNextMoves = None, depth = 1):
        """
            For a given board, given turn and given player
            Return the ensemble of possible moves
            
            Input :
                board : board in matrix representation
                player : actual player
                ignore_check : compute position ignoring the check
                opponentNextMoves : list of opponentNextMoves [useful to compute it only once when verifying if in check]
                depth : int, depth of the function call, used to stop the loop with is_check call
                
            Output :
                move_layers : a list of possible move for each layer
                    Each move is represented by a list of two tuple : previous and new position
        """
        
        board_layers = self._board_to_layer(board)
        
        current_player_positions = self.__player_layers_positions[player]
        current_opponent_positions = self.__player_layers_positions[1-player]
        
        # Only if verifying the check situation
        
        if depth > 2:
            # Prevent check loop : we cannot check with a depth superieur to 2
            ignore_check = True
        
        if ignore_check == False:
            ## Creating the fake board for fast cheacking
            opponentBoard = board.copy() # Board with only the opponent and the king, see below in the check verification
            opponentBoard[(np.mod(opponentBoard, 10) == player) | (opponentBoard != (60+player))] = 0

            opponentNextMoves_noPlayer = self._getNextMove(opponentBoard, player, ignore_check = True)
            
            ## Depth increment
            depth += 1
        
        # Get the positions
        positions = [np.stack(np.where(x == 1)) for x in board_layers]
        player_position = [positions[i] for i in current_player_positions]
        opponent_position = [positions[i] for i in current_opponent_positions]

        moves = {}
        move_functions = {
            0:self._getPionMove,
            1:self._getCavMoves,
            2:self._getBishopMoves,
            3:self._getRookMoves,
            4:self._getLadyMoves,
            5:self._getKingMoves,
            6:self._getRoqueMoves
        }
        
        for i in range(7):
            if move_functions[i] is not None:

                moves_function = move_functions[i](player_position, opponent_position, player)

                if len(moves_function) != 0:
                    temp_move = moves_function.T.reshape(moves_function.shape[1], 2, 2)

                    # We should check if the opponent will make us in check state before considering a move
                    ## To accelerate the pre-compute, we use the opponentNextMoves_noPlayer which is a board without only the opponent and the king, this board is used to filter the move for which it is useful to compute the real situation
                    if ignore_check == False:

                        move_mask = []
                        for current_temp_move in temp_move:
                            temp_board = self._apply_move(current_temp_move, board) # Applying the move to the board
                            move_check = self._is_check(temp_board, player, opponentNextMoves = opponentNextMoves, depth = depth) # Fast checking

                            if move_check == True:
                                move_check = self._is_check(temp_board, player, depth = depth)

                            move_mask.append(
                                (move_check == False)
                            ) # Storing in a mask if the move keep the player in check

                        temp_move = temp_move[move_mask]                        

                    moves[i] = temp_move.tolist()
    
        return moves
    
    def __removeOutOfBorderMoves (self, move_np):
        '''
            For internal purpuse
            Remove out of border moves
            In input : np array
            In output : np array
        '''
        
        if len(move_np) > 0:
            moves = move_np[:,(move_np[2] <= 7) & (move_np[2] >= 0) & (move_np[3] <= 7) & (move_np[3] >= 0)]
        else:
            return move_np
        
        return moves
    
    def __removeCollisionMoves (self, move_np, player_position):
        '''
            For internal purpuse
            Remove out the collision move : moves in the position of current pieces
            In input : np array, player_position
            In output : np array
        '''
        
        if (move_np.shape[1] > 0):
            player_position_concat = np.concatenate(player_position, axis = 1)
            moves_positions = move_np[[2,3],:]
            overlapping_player = np.where((moves_positions[:,None].T == player_position_concat.T).all(axis=2))[0]
            moves = np_delete(move_np, overlapping_player, axis = 1)
        else:
            return move_np
    
        return moves
    
    def _getPionMove (self, player_position, opponent_position, player):
        
        """
            Compute the pion move according to :
                Current player position
                Opponent position
                Current player
                Current turn
                
            Output : list of moves
        """
        
        direction = -(1-player)+(player)*1 # Direction of the move
        all_opponents = np.concatenate(opponent_position, axis = 1) # Location of the opponents
        player_position_concat = np.concatenate(player_position, axis = 1)
        
        # Current position
        wp = player_position[0] # Get the player position
        if len(wp) == 0:
            return []
        
        initial_position = 6 if player == 0 else 1
        wp_repeat = np.concatenate([
            wp for j in range(2)
        ], axis = 1) # Repeat it to cover the possible positions
        
        # 1. Computing classical move : straight forward
        
        npos_1range = np.stack([
            wp[0,:]+direction,
            wp[1,:]
        ]) # Position for move of 1 range

        ## Identification of overlapping
        overlapping_player = np.where((npos_1range[:,None].T == player_position_concat.T).all(axis=2))[0]
        ## Selection of pion for 2 range
        wp_2range_filter = wp[0] == initial_position
        wp_2range_filter[overlapping_player] = False
        wp_2range = wp[:,wp[0] == wp_2range_filter] # Get the player positions which can make two moves
        
        
        npos_2range = np.stack([
            wp_2range[0,:]+2*direction,
            wp_2range[1,:]    
        ]) # Position for moves of 2 range
        
        npos = np.concatenate([npos_1range, npos_2range], axis = 1)
        
        moves = np.concatenate([
            np.concatenate([wp, wp_2range], axis = 1),
            npos
        ])
        
        ## Need to remove position in which there is an opponent
        collisions = np.where((all_opponents[:,None].T == npos.T).all(axis=2))[1]
        #moves = np.delete(moves, collisions, axis = 1)
        moves = np_delete(moves, collisions, axis = 1)
        
        # 2. Computing diagonal moves
        all_diagonal = np.stack([
            np.concatenate([wp[0,:]+direction*(1) for j in list([-1,1])]).flatten(),
            np.concatenate([wp[1,:]+j*(1) for j in list([-1,1])]).flatten()
        ])
        
        possible_moves = np.concatenate([
                wp_repeat,
                all_diagonal
        ])[:,np.where((all_opponents[:,None].T == all_diagonal.T).all(axis=2))[1]]
        
        if possible_moves.shape[-1] != 0:
            moves = np.concatenate([
                moves,
                possible_moves
            ], axis = 1)
        
        # Removing out of board moves
        moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
        
        return moves
    
    def _getCavMoves (self,  player_position, opponent_position, player):
        
        # Current position
        wp = player_position[1] # Get the player position
        wp_repeat = np.concatenate([
            wp for j in range(8)
        ], axis = 1) # Repeat it to cover the possible positions
        
        # Getting new positions
        npos = np.concatenate([wp + self.cav_move[j] for j in range(8)], axis = 1)
        
        # Concatenating with wp
        moves = np.concatenate([
            wp_repeat,
            npos
        ])
        
        moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
        
        return moves
    
    def _getLongRangeMoves (self, rule, wp, player_position, opponent_position, player):
        
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

                # For overlapping player, we disable the move
                temp_pos = np.copy(new_pos)
                if (len(overlapping_player) > 0):
                    temp_pos = np_delete(temp_pos, overlapping_player, axis = 0)

                # Storing the positions
                if len(temp_pos) > 0:
                    new_move = np.concatenate([
                        np.concatenate([pos for n in range(len(temp_pos))], axis = 1),
                        temp_pos.T
                    ])
                    moves.append(new_move)

                # We disable overlapping positions
                overlapping = np.concatenate([overlapping_opponent, overlapping_player])
                
                directions[np.where(directions == True)[0][overlapping]] = False ## In directions
                ## In positions
                if (len(overlapping) > 0):
                    new_pos = np_delete(new_pos, overlapping, axis = 0)
                    
                # We stop after 8 moves
                if n_move == 8:
                    break

                # Go for net loop
                n_move += 1
        
        if (len(moves) > 0):
            moves = np.concatenate(moves, axis = 1)
            moves = self.__removeOutOfBorderMoves(moves)
        
        return moves
        
    def _getBishopMoves (self, player_position, opponent_position, player):
        
        # Current position
        wp = player_position[2] # Get the player position
        moves = self._getLongRangeMoves(self.bishop_move, wp ,player_position, opponent_position, player)
        
        return moves
 
    def _getRookMoves (self,  player_position, opponent_position, player):
        
        # Current position
        wp = player_position[3] # Get the player position
        moves = self._getLongRangeMoves(self.rook_move, wp ,player_position, opponent_position, player)

        return moves
    
    def _getLadyMoves (self, player_position, opponent_position, player):
        
        # Mix of Bishop and Rook, easy one
        wp = player_position[4] # Get the player position
        bishop_moves = self._getLongRangeMoves(self.bishop_move, wp ,player_position, opponent_position, player)
        rook_moves = self._getLongRangeMoves(self.rook_move, wp ,player_position, opponent_position, player)
        
        moves = []
        
        if (len(bishop_moves) != 0 and len(rook_moves) != 0):
            moves = np.concatenate([bishop_moves, rook_moves], axis = 1)
        elif (len(bishop_moves) != 0):
            moves = bishop_moves
        elif (len(rook_moves) != 0):
            moves = rook_moves
        
        return moves
    
    def _getKingMoves (self, player_position, opponent_position, player):
        
        wp = player_position[5] # Get the player position
        moves = []
        
        if wp.shape[1] > 0:
            moves = np.concatenate([
                np.concatenate([wp for i in range(8)], axis = 1),
                self.king_move.T+wp
            ])

            moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
                        
        return moves
    
    def _getRoqueMoves (self, player_position, opponent_position, player):
        
        """
            Compute the roque move according to :
                Current player position
                Opponent position
                Current player
                Current turn
                
            Output : list of moves
        """  
        
        moves = []
        
        all_positions = np.concatenate([
            np.concatenate(player_position, axis = 1),
            np.concatenate(opponent_position, axis = 1)
        ], axis = 1).T.tolist()
        
        if len(all_positions) == 0:
            return []
        
        # Cannot play roque if it has already been
        if self.has_roque[player] == True:
            return []
            
        ## Fast check : not possible if there is someone next to the king
        not_moved = [i[1] for i in self._not_moved[player]] # List of not moved positions
        allow_check_roque = False
        if len(not_moved) > 0:
            y_pos = self._not_moved[player][0][0]
            position_to_check = [[[y_pos, 3], [y_pos, 2]], [[y_pos, 5], [y_pos, 6]]]
            
            for position in position_to_check:
                if (position[0] not in all_positions) and (position[1] not in all_positions):
                    allow_check_roque = True
                    break
        
        ## Full check
        if allow_check_roque == True:
            if 4 in not_moved:
                # Removing the king location
                not_moved.remove(4)
                
                if len(not_moved) > 0:
                    # Compute the Roque for each tower / king combination
                    ## Requirement : there is no one between the king and the tower
                    for tower in not_moved:
                        allow_roque = False # Unless everything is ok, we do not allow the move
                        distance = abs(tower-4)-1 # Get the horizontal distance
                        direction = (1, -1)[(tower-4)<0] # Get the horizontal direction
                        
                        if distance == 2: # Small roque
                            # Checking the absence of position
                            allow_roque = True
                            king_move = 4+2*direction
                            rook_move = tower-2*direction
                            intermediate_move = rook_move # Just to process to one implementation of the check
                        elif distance == 3: # Big roque
                            allow_roque = True
                            king_move = 4+2*direction
                            rook_move = tower-3*direction
                            intermediate_move = tower-(1*direction)

                        # Checking if the move is allowed
                        if allow_roque:
                            if ([y_pos, king_move] not in all_positions) \
                                and ([y_pos, rook_move] not in all_positions) \
                                and ([y_pos, intermediate_move] not in all_positions):
                                moves.append([y_pos, 4, y_pos, king_move])
                                        
        moves = np.array(moves).T
        
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
                
        layer = [(board == x).astype("int8") for x in list(self._inital_board_pieces.keys())]
        
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
    
    def _promotion(self, board, player, promotion = 5):
        
        """
            Generic function for promotion
            Given a board and a player, return the board after application of the promotion
            
            Input :
                - board : matrix representation of the board
                - player : identification of the player
                - the piece in which we want to promote, by default it is 5 (queen)
                
            Return the board
        """
        
        last_row = 7 if (player == 1) else 0
        
        if promotion is not None:
            board[last_row, np.where(
                (np.mod(board[last_row,:], 10) == player) & \
                (((board[last_row,:])-player)/10 == 1)
            )[0]] = (promotion*10)+player
        
        return board
    
    def _apply_move (self, move, board, real = False, promotion = 5):
        
        """
            Basically apply virtually a move to a board without any sanity check
            
            Input :
                move : move to apply
                board : board to apply the move
                real : if true, the move is applied and we check for first move of the king and rook for Roque calculation
            Ouput :
                board after appliance of the move
        """
        
        board_copy = board.copy()
        piece = board_copy[tuple(move[0])] 
        player = piece%10
        
        # Exception : for the roque, we have delete from self._not_moved the piece that moved 
        if real:
            # Fixing move
            if type(move) != type(list()):
                move = move.tolist()
            
            if ((piece-player)/10 in [4,6]):
                if move[0] in self._not_moved[player]:
                    self._not_moved[player].remove(move[0])
        
        board_copy[tuple(move[0])] = 0 # Setting empty the previous position
        board_copy[tuple(move[1])] = piece # Setting the new position           
            
        # If it is a roque, we have to also move the corresponding tower
        if self._is_roque(move, board):
                        
            # Set has roque to True
            if real == True:
                self.has_roque[player] = True

            # Get the corresponding tower move
            if move[1][1] < 4:
                tower_move = [[move[1][0], 0], [move[1][0], 3]]
            else:
                tower_move = [[move[1][0], 7], [move[1][0], 5]]

            # Move the tower
            board_copy = self._apply_move(tower_move, board_copy, real = real)
        
        # Finally, promote pieces
        board_copy = self._promotion(board_copy, player, promotion)
                
        return board_copy
    
    def _is_roque (self, move, board):
        """
            Check if a move is a roque
            
            Input : 
                move : move array
                board : board matrice
            Output : boolean
        """
        
        if type(move) != type(list()):
            move = move.tolist()
        
        initial_position = move[0]
        destination = move[1]
        piece = board[tuple(initial_position)]
        
        is_roque = False
        
        # Determine wether is is a king
        if piece // 10 == 6:
            if initial_position in [[0,4],[7,4]]:
                # Finally determine distance of the move
                is_roque = abs(initial_position[1]-destination[1]) > 1
                
        return is_roque
    
    def play_move(self, move, promotion = 5):
        
        """
            play_move :
                Play a move
                For that :
                    0. Play only if not mate
                    1. Check the validity of the move
                    2. Play the move
                    3. Storing and recomputing the new state
                    4. Update the player and eventually the turn
                    5. Evaluate if there is a winner
                    
                Input :
                    Move : format [[x_source,y_source], [x_target, y_targer]]
                    Promotion : id of the piece to which to promote in case of promotion
        """
        
        # 0. Check if mate
        if self.mate == True:
            return
        
        # 1. Check the move
        ## Getting possible moves
        if move not in self.nextMoves:
            raiseException("incorrect_move")
        
        # 2. Play the move
        board_copy = self._apply_move(move, self.board, real = True, promotion = promotion)
        self.board = self._apply_move(move, self.board, real = True, promotion = promotion)

        # 4. Storing and recomputing the new state
        self._memorize_board(self.board)
        
        # 5. Update the player and eventually the turn
        self.current_player = 1-self.current_player
        if self.current_player != 0:
            self.turn += 1
        
        # 6. Evaluate if there is a winner
        ## Get next moves
        self.nextMoves = self._getCurrentNextMove()
        self.opponentNextMoves = self._getOpponentsCurrentNextMove()
        
        ## Checking if in check
        self.check = self.is_check()
        if (self.check):
            self.mate = self.is_mate()
            self.winner = 1-self.current_player
        
        pass
    
    
    def play_move_sf (self, move, promotion = 5):
        
        """
            play_move :
                Play a move from a stockFish format
                
                Input :
                    Move :
                        - Move in stockfish format
                    Promotion : id of the piece to which to promote in case of promotion
        """
        
        first_move = (self.first_player == 0)
        move = self._stockFishToMove(move, first_move)
        
        self.play_move(move, promotion)
        
        pass
        
    ###
    # Ensemble of functions for game evaluation
    ###
    
    def _is_check(self, board, player, opponentNextMoves = None, depth = 1):
        
        """
            Compute for a board configuration if a player is in a check situation
            
            Input :
                board : matrix of the current board
                player : id of the player
                opponentNextMoves : list of opponentNextMoves [useful to compute it only once], if not given it is computed
                depth : depth of the function call, used to stop the loop with getNextMove
                
            Return : boolean of check situation
        """
        
        king_location = np.where(board == 60+player)
        ignore_check = False
        
        # Compute it if not given
        if opponentNextMoves is None:
            if depth > 1:
                ignore_check = True
            opponentNextMoves = self._getNextMove(board, 1-player, ignore_check = ignore_check, depth = depth)
            opponentNextMoves = reduce(add, list(opponentNextMoves.values())) # Set it to a list
            
        if king_location[0].sum() > 0:
            king_location = np.array(king_location).reshape(2).tolist()
            check_situations = [True for x in opponentNextMoves if king_location == x[1]]
            is_check = len(check_situations) > 0
        else:
            is_check = False
        
        return is_check
    
    def is_check(self):
        
        """
            Analyze is the current game is check
            
            Input : None
            Output : boolean, true if check, false otherwise
        """
        
        is_check = self._is_check(self.board, self.current_player, self.opponentNextMoves)
        
        return is_check
    
    def is_mate(self):

        """
            Analyze is the current game is check and mate
            
            Input : None
            Output : boolean, true if check and mate, False otherwise
        """
        
        # Easy one : it is check and mate if it is check with no possible next move
        mate = len(self.nextMoves) == 0
        
        return mate
    
    ###
    # Ensemble of functions to display the board
    ###
    
    def _get_board (self, board, cell_size = 3, representation = None):
        
        """
            Display a given board
            Input :
                board : matrix of the board to display
                cell_size : size of each cell of the board
                representation : representation mode of the board, if None the default one is use (coordonates between 0 and 7), if "sf", the Stockfish one is used
            Output :
                String : representation of the board
        """
        
        # Make sure to have numpy array on cpu
        board = np_cpu.array(board.tolist())

        # Getting the coordonates
        if representation == 'sf':
            coordonates_x = self.board_coordonates["sf"]["x"]
            coordonates_y = self.board_coordonates["sf"]["y"]            
        else:
            coordonates_x = self.board_coordonates["default"]["x"]
            coordonates_y = self.board_coordonates["default"]["y"]
        
        # New board
        
        visual_board = board
        visual_board = visual_board.astype("U2") # Converting in string
        visual_board[visual_board == ["0"]] = "" # The empty space will be empty

        # Creating a board with letter instead of integer
        for i in range(2):
            
            # Piece filter
            pieces = (np_cpu.mod(board, 10) == i) & (board != 0)
            player_pieces = np_cpu.divide((board[pieces]-i), 10).astype("int8")

            visual_board[pieces] = self.board_dictionnary[player_pieces-1]
            if i == 0:
                visual_board[pieces] = np_cpu.char.capitalize(visual_board[pieces])

        # Generating the string
        visual_board = np_cpu.concatenate([
            coordonates_y,
            visual_board
            ], axis = 1
        )

        board_string = ""
        line_separator = "".join(["-" for k in range(8*(cell_size+1))])
        cell_gap = "".join([" " for k in range(cell_size)])

        board_string += ("\n"+cell_gap+line_separator+"\n")
        board_string += ("\n"+cell_gap+line_separator+"\n").join(["|".join([y.center(cell_size) for y in x])+"|" for x in visual_board.tolist()])
        board_string += ("\n"+cell_gap+line_separator+"\n")
        board_string += (cell_gap+" ")+" ".join([x.center(cell_size) for x in coordonates_x[0].tolist()])

        return(board_string)
        
    def get_board(self, cell_size = 3, representation = None):
        
        """
            Display the current board
            Input :
                cell_size : size of each cell of the board
                representation : representation mode of the board, if None the default one is use (coordonates between 0 and 7), if "sf", the Stockfish one is used
            Output :
                String : representation of the board        
        """
        
        return self._get_board(board = self.board, cell_size = cell_size, representation = representation)


