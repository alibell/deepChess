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
import copy


# +
# Convention :
## 0 : Is white, 1 is black
## 1 : Pawn, 2 : Knight, 3 : Bishop, 4 : Rook, 5 : Queen, 6 : King
## The player is always white and always in the bottom of the matrix
## Example : 21 is a black knight, 0 is the absence of piece
### Thus, we store a full chess board with a 8x8 matrix containing the 64 8-bit integer : 512 bit
# -

##
# Functions for board manipulation
##

def _np_delete (np_array, list_to_remove, axis = 0):
    
    """
        Faster implementation of np.remove, do it by getting the right index
    """
    
    if (len(list_to_remove) > 0):


        list_of_indexes = np.array(list(range(np_array.shape[axis])))
        mask = list_of_indexes[(np.isin(list_of_indexes, np.array(list_to_remove)) == False)]

        np_array = np.take(np_array, mask, axis = axis)
    
    return np_array

##
# Pre-computing transcodage rules
##

# Generating the moves identifiant and the moves matrice
## Needed for transcodage of moves in neural network format

def _generate_moves_matrice ():
    distance = list(range(1, 8, 1)) # Distance of action
    direction = ["NW","N","NE","E","SW","S","SE","W"] # Direction of action
    action = ["S","K","PK","PB","PR"] # Types of action

    moves = {}
    moves_matrice = np.zeros((len(distance), len(direction), len(action)))

    move_index = 0
    for i in range(len(direction)):
        for j in range(len(distance)):
            for k in range(len(action)):
                _action = action[k]
                _direction = direction[i]
                _distance = int(distance[j])

                # Do not process moves > 1 when not going straightforward
                if _action != 'S' and _distance > 1:
                    break
                # Process promote only if pion
                if _action[0] == 'P' and (_direction not in ['N','NW','NE']):
                    if _direction in ["S","SW","SE"]:
                        _direction = ['N','NW','NE'][["S","SW","SE"].index(_direction)] # Same id in P for S, SW and SE
                    else:
                        break

                # Store the move id
                move_id = _action+"_"+_direction+"_"+str(_distance)
                if move_id not in moves.keys():
                    moves[move_id] = move_index
                    move_index += 1

                # Store the move id
                moves_matrice[j,i,k] = moves[move_id]+1

    moves_matrice = moves_matrice.astype("int8")
    
    # Pre-computation of matrice for direction
    directions_matrice_ref = np.zeros((2,2,2,2)) # S E N W

    for i in range(len(direction)):
        coordonates = tuple((int(j in direction[i]) for j in ['S','E','N','W']))
        directions_matrice_ref[coordonates] = i+1

    directions_matrice_ref = directions_matrice_ref.astype("int8")
    promote_dictionnary_ref = {2:'K',3:'B',4:'R'}
    
    return moves, moves_matrice, direction, directions_matrice_ref

moves_ref, moves_matrice_ref, directions_ref, directions_matrice_ref = _generate_moves_matrice()

##
# Main class
##

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
        self.previousMove = None
        
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
        
        ## Board hash : count every board configure except for initial position that cannot be repeted
        self._board_hash_history = {}

        ## Memorize the absence of progress
        self._no_progress = 0

        ## Sending it to board history
        self._memorize_board(self.board)
                
        ## Pre-compute next moves
        self.opponentNextMoves = self._getOpponentsCurrentNextMove()
        self.nextMoves = self.getCurrentNextMove()
    
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
    
    def getCurrentNextMove (self):
        
        """
            For the actual board, return the possible next moves
            Output : list composed of next moves.
                     Each move is composed of a list of source coordonates and target coordonates
        """
        
        moves = self._getNextMove(self.board, self.current_player, previousMove=self.previousMove, opponentNextMoves=self.opponentNextMoves, ignore_check=False, depth=1)
        moves_list = reduce(add, list(moves.values()))
        
        return moves_list
    
    def getCurrentNextMoveWithNN (self):
        
        """
            For the actual board, return the possible next moves with the NN representation of the moves
            Output :
                moves_list : list composed of next moves. : Each move is composed of a list of source coordonates and target coordonates
                moves_matrice : matrice of 8x8x73, where the 8x8 plane represente each movable piece and the 73 dimension represention each possible move, as used in the representation of the moves in the neural network
        """
        
        moves_list = self.getCurrentNextMove()
        nn_moves = self._localToNNMove(moves_list, self.board, self.current_player)
        
        return moves_list, nn_moves[1], nn_moves[0]
    
    def _getOpponentsCurrentNextMove (self):
        
        """
            For the actual board, return the possible next moves of the opponents (1 move in the futur)
            Output : list composed of next moves.
                     Each move is composed of a list of source coordonates and target coordonates
        """
        
        moves = self._getNextMove(self.board, 1-self.current_player, previousMove = None)
        moves_list = reduce(add, list(moves.values()))
        
        return moves_list
    
    ##
    # Correspondance between local and stockFish moves
    ##
    
    def _movesToStockFish (self, moves, first_move):
        """
            Convert a list of local moves to a list of stockFish move
            
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
    
    ##
    # Correspondance between local and DeepCNN move
    ##
    
    def _localToNNMove (self, moves, board, player):
    
        """
            localToMove : Convert a local move to the neural network output format
                The neural network output format is of size 8x8x73, with 8x8 representing the 64 possible concerned piece and 73 the possible move
                Each value which is set to 1 correspond to a possible move
                
                Input :
                    moves : moves to convert, in local format
                    board : board as a matrix representation
                    player : current player
                Ouput :
                    moves_id : list containing the id of each move
                    moves_matrice : moves matrice in neural network format, of size 8x8x73
        """
        
        # Just to not rewrite everything
        next_move = moves
        current_player = player
        promote_rank = 0 if current_player == 0 else 7 # The promote rank is needed to identify when a promotion could happen

        #
        # Converting move list to list of moves index
        #

        next_move_array = np.array(next_move).reshape((len(next_move),4)) # Converting to numpy array

        pieces = (board[next_move_array[:,0], next_move_array[:,1]]//10) # Type of pieces

        # Pre-computing the features
        ## Each move is represented by 3 features :
        ##   - Direction : N, NE, NW, E, SE, S, SW, W. For bishop an equivalent representation is used based on the direction of the longuest part of the move (of distance 2) and the sens of the shortest part
        ##   - Distance : distance between start point and end, between 1 and 7, by convention it is set to 1 unless the action (see bellow) is straighforward
        ##   - Action : type of move, we distinguish 5 types : straightforward (s) which is a standard move in one of the 8 direction, knight move (k) and unusual promote move (bishop, knight or rook
        ## With theses 3 characteristic, we can get the move in the moves_matrice_ref, the move id correspond to the id registered in the moves_ref dictionnary
        
        difference = next_move_array[:, [0,1]]-next_move_array[:, [2,3]]
        distance = np.sqrt(np.square(difference).sum(axis = 1)).round(2)
        is_k = (distance == 2.24)
        is_promote = ((pieces == 1) & (next_move_array[:,2] == promote_rank)).astype("int8")

        # Getting move directions and sens
        move_direction = (next_move_array[:, [0,1]] != next_move_array[:, [2,3]])
        move_sens = np.sign(difference[:,[0,1]])
        ## Conversion of knight move to std move
        move_direction_k = (np.abs(difference[is_k]) == 2) \
                            | ((np.abs(difference[is_k]) == 1) & (np.sign(difference[is_k]) == 1))
        move_direction[is_k] = move_direction_k

        # +
        # Getting matrice indices
        _move_direction_sens = move_sens*move_direction.astype("int")
        _direction_matrice_indices = np.concatenate([
            (_move_direction_sens > 0).astype("int"),
            (_move_direction_sens < 0).astype("int")
        ], axis = 1)

        direction_indices = directions_matrice_ref[_direction_matrice_indices[:,0], _direction_matrice_indices[:,1], _direction_matrice_indices[:,2], _direction_matrice_indices[:,3]]-1
        action_indices = (is_promote*pieces)+(1*is_k.astype("int8"))
        is_s = (action_indices == 0).astype("int8")
        distance_indices = (((is_s*distance)+(1-is_s))-1).astype("int")

        # Getting moves coordonates
        moves_id = moves_matrice_ref[distance_indices, direction_indices, action_indices]-1

        moves_matrice = np.zeros((8,8,len(moves_ref)))
        
        moves_matrice[
            next_move_array[:,0],
            next_move_array[:,1],
            moves_id
        ] = 1
        
        return moves_id, moves_matrice
    
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
        
        board_layers = self._board_to_layer(board)
        
        current_player_positions = self.__player_layers_positions[player]
        current_opponent_positions = self.__player_layers_positions[1-player]
        
        # Only if verifying the check situation
        
        if depth > 2:
            # Prevent check loop : we cannot check with a depth superieur to 2
            ignore_check = True
        
        if ignore_check == False:
            ## Creating the fake board for fast cheacking
            opponentBoard = board.copy() # Board with only the opponent pieces and the king, see below in the check verification
            opponentBoard[(np.mod(opponentBoard, 10) == (player)) & ((opponentBoard // 10) != 6)] = 0
            opponentNextMoves_noPlayer = self._getNextMove(board = opponentBoard, player = 1-player, previousMove = previousMove, opponentNextMoves = [], ignore_check = True, alwaysDiag = True)
            opponentNextMoves_noPlayer = reduce(add, list(opponentNextMoves_noPlayer.values()))

            ## Depth increment
            depth += 1
        
        # Setting default previousMove
        if previousMove is None:
            previousMove = self.previousMove

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
                if i == 6:
                    moves_function = move_functions[i](player_position = player_position, opponent_position = opponent_position, previousMove = previousMove, player = player, opponentNextMoves = opponentNextMoves) # We need the opponentNextMoves here
                else:
                    moves_function = move_functions[i](player_position = player_position, opponent_position = opponent_position, previousMove = previousMove, player = player, alwaysDiag = alwaysDiag) # We need the opponentNextMoves here

                if len(moves_function) != 0:
                    temp_move = moves_function.T.reshape(moves_function.shape[1], 2, 2)

                    # We should check if the opponent will make us in check state before considering a move
                    ## To accelerate the pre-compute, we use the opponentNextMoves_noPlayer which is a board without only the opponent and the king, this board is used to filter the move for which it is useful to compute the real situation
                    if ignore_check == False:

                        move_mask = []
                        for current_temp_move in temp_move:
                            temp_board = self._apply_move(current_temp_move, board) # Applying the move to the board
                            move_check = self._is_check(board=temp_board, player=player, opponentNextMoves = opponentNextMoves_noPlayer, previousMove = previousMove, depth = depth) # Fast checking

                            if move_check == True:
                                move_check = self._is_check(board=temp_board, player=player, previousMove = previousMove, opponentNextMoves = None, depth = depth)

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

    def _get_collision(self, a, b):
        """
            From two array identifie the lines with the same value in the two arrays
            
            Input :
                a : first array,
                b : second array
            Output :
                tuple with position in array a and position in array b
        """
        
        res0, res1 = tuple(np.where((a[:,None].T == b.T).all(axis=2)))
        
        return res0, res1

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
            moves = _np_delete(move_np, overlapping_player, axis = 1)
        else:
            return move_np
    
        return moves
    
    def _getPionMove (self, player_position, opponent_position, previousMove, player, alwaysDiag = False):
        
        """
            Compute the pion move according to :
                Current player position
                Opponent position
                Current player
                Current turn
                alwaysDiag : by default False, if true, the pion diagonal is always possible (use for fast checking of king check). By extension, is disable the remove collision filter.
                
            Output : list of moves
        """
        
        direction = -(1-player)+(player)*1 # Direction of the move
        all_opponents = np.concatenate(opponent_position, axis = 1) # Location of the opponents
        player_position_concat = np.concatenate(player_position, axis = 1)
        all_positions = np.concatenate([
            player_position_concat,
            all_opponents
        ], axis = 1)        
        
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

        ## Selection of pion for 2 range
        wp_2range_filter = (wp[0] == initial_position)
        wp_2range = np.array([wp[0]+direction, wp[1]])[:, wp_2range_filter]
        wp_2range = _np_delete(wp_2range, np.where((wp_2range[:,None].T == all_positions.T).all(axis = 2))[0], axis = 1)
        wp_2range[0] -= direction # Going back to initial position (to not break the rest of the code)

        npos_2range = np.stack([
            wp_2range[0,:]+(2*direction),
            wp_2range[1,:]    
        ]) # Position for moves of 2 range
        npos = np.concatenate([npos_1range, npos_2range], axis = 1)
        
        moves = np.concatenate([
            np.concatenate([wp, wp_2range], axis = 1),
            npos
        ])
        
        ## Need to remove position in which there is an opponent
        collisions = np.where((all_opponents[:,None].T == npos.T).all(axis=2))[1]
        moves = _np_delete(moves, collisions, axis = 1)
        
        # 2. Computing eating "en passant"
        passant_moves = np.array([[],[],[],[]]) # Empty moves unless there is one
        if previousMove is not None:
            if (previousMove[0] == 1-player) \
                and (previousMove[2]//10 == 1) \
                and (abs(previousMove[1][1][0]-previousMove[1][0][0]) == 2):

                opponent_passant = np.array([previousMove[1][1]]).T
                player_passant = wp[:, wp[0] == opponent_passant[0,0]]

                # Searching for opponent
                passant_moves = []
                for i in [-1, 1]:
                    passant_collisions = self._get_collision((player_passant + np.array([[0,i]]).T), opponent_passant)
                    
                    # Recording the move
                    passant_moves.append(
                        np.concatenate([
                            player_passant[:,passant_collisions[0]],
                            player_passant[:,passant_collisions[0]]+np.array([[direction,i]]).T,
                        ])
                    )
                passant_moves = np.concatenate(passant_moves, axis = 1)

        # 3. Computing diagonal moves
        all_diagonal = np.stack([
            np.concatenate([wp[0,:]+direction*(1) for j in list([-1,1])]).flatten(),
            np.concatenate([wp[1,:]+j*(1) for j in list([-1,1])]).flatten()
        ])
        
        if alwaysDiag == False:
            possible_moves = np.concatenate([
                    wp_repeat,
                    all_diagonal
            ])[:,np.where((all_opponents[:,None].T == all_diagonal.T).all(axis=2))[1]]
        else:
            possible_moves = np.concatenate([
                    wp_repeat,
                    all_diagonal
            ])

        # 4. Mixing all together
        all_moves = [x for x in [moves, possible_moves, passant_moves] if x.shape[-1] != 0]
        if len(all_moves) > 0:
            moves = np.concatenate(all_moves, axis = 1)
        
        # Removing out of board moves
        if alwaysDiag == False:
            moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
        
        return moves
    
    def _getCavMoves (self,  player_position, opponent_position, previousMove, player, alwaysDiag):
        
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
        
        if alwaysDiag == False:
            moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
        
        return moves
    
    def _getLongRangeMoves (self, rule, wp, player_position, opponent_position, previousMove, player, alwaysDiag = False):
        
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
                if alwaysDiag == False:
                    if (len(overlapping_player) > 0):
                        temp_pos = _np_delete(temp_pos, overlapping_player, axis = 0)

                # Storing the positions
                if len(temp_pos) > 0:
                    new_move = np.concatenate([
                        np.concatenate([pos for n in range(len(temp_pos))], axis = 1),
                        temp_pos.T
                    ])
                    moves.append(new_move)

                # We disable overlapping positions
                if alwaysDiag == False:
                    overlapping = np.concatenate([overlapping_opponent, overlapping_player])
                    
                    directions[np.where(directions == True)[0][overlapping]] = False ## In directions
                    ## In positions
                    if (len(overlapping) > 0):
                        new_pos = _np_delete(new_pos, overlapping, axis = 0)
                    
                # We stop after 8 moves
                if n_move == 8:
                    break

                # Go for net loop
                n_move += 1
        
        if (len(moves) > 0):
            moves = np.concatenate(moves, axis = 1)
            if alwaysDiag == False:
                moves = self.__removeOutOfBorderMoves(moves)
        
        return moves
        
    def _getBishopMoves (self, player_position, opponent_position, previousMove, player, alwaysDiag = False):
        
        # Current position
        wp = player_position[2] # Get the player position
        moves = self._getLongRangeMoves(self.bishop_move, wp ,player_position, opponent_position, previousMove, player, alwaysDiag)
        
        return moves
 
    def _getRookMoves (self,  player_position, opponent_position, previousMove, player, alwaysDiag = False):
        
        # Current position
        wp = player_position[3] # Get the player position
        moves = self._getLongRangeMoves(self.rook_move, wp ,player_position, opponent_position, previousMove, player, alwaysDiag)

        return moves
    
    def _getLadyMoves (self, player_position, opponent_position, previousMove, player, alwaysDiag = False):
        
        # Mix of Bishop and Rook, easy one
        wp = player_position[4] # Get the player position
        bishop_moves = self._getLongRangeMoves(self.bishop_move, wp ,player_position, opponent_position, previousMove, player, alwaysDiag)
        rook_moves = self._getLongRangeMoves(self.rook_move, wp ,player_position, opponent_position, previousMove, player, alwaysDiag)
        
        moves = []
        
        if (len(bishop_moves) != 0 and len(rook_moves) != 0):
            moves = np.concatenate([bishop_moves, rook_moves], axis = 1)
        elif (len(bishop_moves) != 0):
            moves = bishop_moves
        elif (len(rook_moves) != 0):
            moves = rook_moves
        
        return moves
    
    def _getKingMoves (self, player_position, opponent_position, previousMove, player, alwaysDiag = False):
        
        wp = player_position[5] # Get the player position
        moves = []
        
        if wp.shape[1] > 0:
            moves = np.concatenate([
                np.concatenate([wp for i in range(8)], axis = 1),
                self.king_move.T+wp
            ])

            if alwaysDiag == False:
                moves = self.__removeCollisionMoves(self.__removeOutOfBorderMoves(moves), player_position)
                        
        return moves
    
    def _getRoqueMoves (self, player_position, opponent_position, previousMove, player, opponentNextMoves):
        
        """
            Compute the roque move according to :
                Current player position
                Opponent position
                Current player
                Current turn
                
            Output : list of moves
        """  
        
        moves = []
        
        if opponentNextMoves is not None:
            opponentNextMoves = [x[1] for x in opponentNextMoves]
        else:
            opponentNextMoves = []

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

            ## Another fast check : not possible is the king is check 
            if [y_pos,4] in opponentNextMoves:
                allow_check_roque = False

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

                        # Checking if allowed
                        if ([y_pos, rook_move] in opponentNextMoves) or ([y_pos, king_move] in opponentNextMoves):
                            allow_roque = False

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
    
    def _board_to_NN_input(self, board_layers, turn, player, hash_history, not_moved, no_progress):
        """
            Given an game configuration, it generate the input for the Neural Network. The input is composed of :
                A stack of matrice for pieces positions
                A vector for additional information : player, the turn, the number of repetitions, the number of castling, the number of move without progress
            Input :
                board_layers : list of layers of the board to analyse
                turn : int, number of turn
                player : int, identifiant of the player, 0 or 1
                hash_history : dict, dictionnary containing the hash of all position and the number of occurrence
                not_move : dict, dictionnary containing the move situation pour roque related pieces
                no_progress : int, number of moves without any progress (eaten piece)
            Output :
                board_layers_for_nn : numpy stack of matrice representing the location of all the pieces during the last turns
                features_for_nn : vector of features containing the scalar features
        """
        
        n_layers = 8 # Number of layers to include in the layer representation of the board
        king_row = {
            0:7,
            1:0
        } # Row containing the king, for analyze of the not moved pieces

        n_hash_history = len(hash_history.values())
        if n_hash_history > 0:
            repeats = copy.deepcopy(list(hash_history.values()))
            repeats.sort(reverse = True)

            n_repeat = repeats[0:5]
            if (len(n_repeat) < 5):
                for i in range(5-len(n_repeat)):
                    n_repeat.append(0)
        else:
            n_repeat = [0 for i in range(5)]

        # Positions layers
        board_layers = [np.stack(x) for x in board_layers]
        board_layers_for_nn = board_layers[-n_layers:-1]+[board_layers[-1]]

        l_board_layers = len(board_layers_for_nn)
        if l_board_layers < 8:
            missing_layers = 8-l_board_layers
            board_layers_for_nn += [np.zeros(board_layers_for_nn[-1].shape, dtype = "int8") for i in range(missing_layers)]
        board_layers_for_nn = np.concatenate(board_layers_for_nn)

        # Scalar features
        castling = []
        for i in [0,1]:
            if [king_row[i],4] in not_moved[i]:
                castling.append(len(not_moved[i])-1)
            else:
                castling.append(0)

        features_for_nn = [player, turn, *n_repeat, *castling, no_progress] 
        
        return board_layers_for_nn, features_for_nn
    
    def current_board_to_NN_input(self):
        """
            Given an game configuration, it generate the input for the Neural Network. The input is composed of :
                A stack of matrice for pieces positions
                A vector for additional information : player, the turn, the number of repetitions, the number of castling, the number of move without progress
            Input :
                None
            Output :
                board_layers_for_nn : numpy stack of matrice representing the location of all the pieces during the last turns
                features_for_nn : vector of features containing the scalar features
        """

        board_layers = self.board_layers_history
        turn = self.turn
        player = self.current_player
        not_moved = self._not_moved
        no_progress = self._no_progress
        hash_history = self._board_hash_history

        res = self._board_to_NN_input(board_layers, turn, player, hash_history, not_moved, no_progress)
        return res
    
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
        piece_type = piece//10

        # Exception 1 : for the roque, we have delete from self._not_moved the piece that moved 
            # Also, we have to remove from not_moved the piece which are eaten
        if real:
            # Fixing move
            if type(move) != type(list()):
                move = move.tolist()
            
            if (piece_type in [4,6]):
                if move[0] in self._not_moved[player]:
                    self._not_moved[player].remove(move[0])
            
            # Removing eaten piece
            if move[1] in self._not_moved[1-player]:
                self._not_moved[1-player].remove(move[1])

        # Exception 3 : If it is a "prise au passant", we should to get ride of the opponent
        if piece_type == 1 and (move[0][1] != move[1][1]) and board_copy[tuple(move[1])] == 0: # If we are moving, to and empty direction, in diagonal : it is prise au passant
            board_copy[move[0][0], move[1][1]] = 0

        board_copy[tuple(move[0])] = 0 # Setting empty the previous position
        board_copy[tuple(move[1])] = piece # Setting the new position
            
        # Exception 2: If it is a roque, we have to also move the corresponding tower
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
                    5. Compute the number of same board configuration
                    6. Evaluate if there is a winner or a draw
                    
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
            raiseException("incorrect_move", move)
        piece = self.board[tuple(move[0])] 
        
        # 2. Play the move
        # board_copy = self._apply_move(move, self.board, real = True, promotion = promotion) # don't know why it is here, maybe I should remove it
        self.board = self._apply_move(move, self.board, real = True, promotion = promotion)

        # 3. Storing and recomputing the new state
        self._memorize_board(self.board)
        self.previousMove = (self.current_player, move, piece) # Storing the previous move
        
        # 4. Update the player and eventually the turn
        self.current_player = 1-self.current_player
        
        if self.current_player == self.first_player:
            self.turn += 1
            
        ## Update the no progress counter
        if (piece//10 != 1) and (self.board_history[-2] != 0).sum() == (self.board_history[-1] != 0).sum():
            self._no_progress += 1
        else:
            self._no_progress = 0
        
        # 5. Count the number of repetition of each board configuration
        _board_hash = hash(self.board.flatten().tostring())
        if _board_hash in self._board_hash_history.keys():
            self._board_hash_history[_board_hash] += 1
        else:
            self._board_hash_history[_board_hash] = 1
        
        # 6. Evaluate if there is a winner
        
        ## Get next moves
        self.nextMoves = self.getCurrentNextMove()
        self.opponentNextMoves = self._getOpponentsCurrentNextMove()
        
        ## Checking if in check
        self.check = self.is_check()
        if (self.check):
            self.mate = self.is_mate()
            self.winner = 1-self.current_player
            
        ## Check if in draw
                
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
        
        king_location = np.where(board == 60+player)
        king_location = np.array(king_location)
        ignore_check = False
        
        # Compute it if not given
        if opponentNextMoves is None:
            if depth > 1:
                ignore_check = True
            opponentNextMoves = self._getNextMove(board, 1-player, ignore_check = ignore_check, previousMove = previousMove, depth = depth)
            opponentNextMoves = reduce(add, list(opponentNextMoves.values())) # Set it to a list
        
        if (king_location.shape[1]) > 0:
            king_location = king_location.reshape(2).tolist()
            check_situations = [True for x in opponentNextMoves if king_location == x[1]]
            is_check = len(check_situations) > 0
        else:
            is_check = False
        
        return is_check
    
    def is_check(self):
        
        """
            Analyze if the current game is check
            
            Input : None
            Output : boolean, true if check, false otherwise
        """
        
        is_check = self._is_check(board = self.board, player = self.current_player, opponentNextMoves = self.opponentNextMoves, previousMove =self.previousMove)
        
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

    def is_draw(self):

        """
            Analyse if the current game is draw

            Input : None
            Output : boolean, true if draw, False otherwiser
        """

        draws = []
        n_next_moves = len(self.nextMoves) # Number of possible next moves
        board = self.board[self.board != 0]
        pieces = board // 10
        players = board % 10
        players_pieces = [pieces[(players == i)].tolist() for i in [0,1]]

        # Check the number of possible moves
        if n_next_moves == 0:
            draws.append("pat")

        # Check the pieces in game
        if len(pieces) < 4:
            for i in players_pieces:
                if players_pieces[i] == [6]: # If there is only the king
                    if players_pieces[1-i] in [[6, 3], [3,6], [6], [6,2], [2,6]]:
                        draws.append("dead")
                elif players_pieces[i] in [[6,3],[3,6]]:
                    if players_pieces[1-i] in [[6, 3], [3,6]]:
                        draws.append("dead")
        
        # Check if the same position occured 3 times
        if len([i for i in self._board_hash_history.values() if i >= 3]):
            draws.append("same_3")

        # Check if there was no progress
        if self._no_progress >= 50:
            draws.append("50_moves")

        if len(draws) > 0:
            is_draw = True
        else:
            is_draw = False
   
        return is_draw, draws


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

    """
        Function to export the board
    """

    def get_fen_position(self):

        # Getting variables
        current_player = self.current_player
        first_move = self.first_player
        previousMove = self.previousMove
        no_progress = self._no_progress
        not_moved = self._not_moved
        turn = self.turn
        direction = -(1-current_player)+(current_player)*1 # Direction of the move
        
        # Color dictionnary : the first player is white
        color = {first_move:'w', 1-first_move:'b'}

        # New board
        visual_board = self.board.copy()
        visual_board = visual_board.astype("U2") # Converting in string
        visual_board[visual_board == ["0"]] = "" # The empty space will be empty

        # Creating a board with letter instead of integer
        for i in range(2):
            # Piece filter
            pieces = (np_cpu.mod(self.board, 10) == i) & (self.board != 0)
            player_pieces = np_cpu.divide((self.board[pieces]-i), 10).astype("int8")

            visual_board[pieces] = self.board_dictionnary[player_pieces-1]
            
            if i == first_move:
                visual_board[pieces] = np_cpu.char.capitalize(visual_board[pieces])

        # Reversing the board according to first player
        if first_move == 1:
            visual_board = np.flip(visual_board, axis = 0)
        
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
        color = color[current_player]

        # Getting castle state
        castles = {
            0:'',
            1:''
        }
        for i in [0,1]:
            castles_positions = [x[1] for x in not_moved[i]]
            if 4 in castles_positions:
                if 7 in castles_positions:
                    castles[i] += 'K'
                if 0 in castles_positions:
                    castles[i] += 'Q'
                    
        if castles[0] == '' and castles[1] == '':
            castles[0] = '-'
            

        # Getting en passant move
        if  (previousMove is not None) \
            and (previousMove[2]//10 == 1) \
            and (abs(previousMove[1][0][0]-previousMove[1][1][0]) == 2):
            move_to_encode = copy.deepcopy(previousMove[1])
            move_to_encode[1][0] += direction
            en_passant = self._movesToStockFish([move_to_encode], 1-first_move)[0][2:]
        else:
            en_passant = "-"

        fen_string = '/'.join(fen_list)
        fen_string += f' {color}'
        fen_string += f' {castles[first_move]+castles[1-first_move].lower()}'
        fen_string += f' {en_passant}'
        fen_string += f' {no_progress} {turn}'

        return fen_string


