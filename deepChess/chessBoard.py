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
    
    def __init__ (self, load_state = None):
        """
            load_state :
                None if the chess is initialized as a new game
                (board, turn) : if the chess is initialized from a current position, then board is the board matrix and turn a binary integer representing the next player
        """
        
        # Knownleage about chess

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
                        
        ## Initialization of number of turn
        self.turn = 1
        self.mate = False # Play until is a mate

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
    
    def _getCurrentNextMove (self):
        
        """
            For the actual board, return the possible next moves
            Output : list composed of next moves.
                     Each move is composed of a list of source coordonates and target coordonates
        """
        
        moves = self._getNextMove(self.board_layers_history[-1], self.current_player, self.turn)
        moves_list = reduce(add, list(moves.values()))
        
        return moves_list
    
    def _getOpponentsCurrentNextMove (self):
        
        """
            For the actual board, return the possible next moves of the opponents (1 move in the futur)
            Output : list composed of next moves.
                     Each move is composed of a list of source coordonates and target coordonates
        """
        
        moves = self._getNextMove(self.board_layers_history[-1], 1-self.current_player, self.turn+1)
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
        
        current_player_positions = self.__player_layers_positions[player]
        current_opponent_positions = self.__player_layers_positions[1-player]

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
                moves_function = move_functions[i](player_position, opponent_position, player, turn)
                if len(moves_function) != 0:
                    moves[i] = moves_function.T.reshape(moves_function.shape[1], 2, 2).tolist()
            
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
        all_oponents = np.concatenate(opponent_position, axis = 1) # Location of the opponents
        
        # Current position
        wp = player_position[0] # Get the player position
        wp_repeat = np.concatenate([
            wp for j in range(2)
        ], axis = 1) # Repeat it to cover the possible positions
        
        wp_range = wp_repeat if current_range == 2 else wp
        
        # 1. Computing classical move : straight forward
        
        npos = np.stack([
            np.concatenate([wp[0,:]+direction*(j+1) for j in range(current_range)]).flatten(),
            wp_range[1,:]
        ])
        
        moves = np.concatenate([
            wp_range,
            npos
        ])
        
        ## Need to remove position in which there is an opponent
        collisions = np.where((all_oponents[:,None].T == npos.T).all(axis=2))[1]
        moves = np.delete(moves, collisions, axis = 1)

        
        # 2. Computing diagonal moves
        all_diagonal = np.stack([
            np.concatenate([wp[0,:]+direction*(1) for j in list([-1,1])]).flatten(),
            np.concatenate([wp[1,:]+j*(1) for j in list([-1,1])]).flatten()
        ])
        
        possible_moves = np.concatenate([
                wp_repeat,
                all_diagonal
        ])[:,np.where((all_oponents[:,None].T == all_diagonal.T).all(axis=2))[1]]
        
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
        npos = np.concatenate([wp + self.cav_move[j] for j in range(8)], axis = 1)
        
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

                # For overlapping player, we disable the move
                temp_pos = np.copy(new_pos)
                if (len(overlapping_player) > 0):
                    temp_pos = np.delete(temp_pos, overlapping_player, axis = 0)

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
                    new_pos = np.delete(new_pos, overlapping, axis = 0)
                    
                # We stop after 8 moves
                if n_move == 8:
                    break

                # Go for net loop
                n_move += 1
        
        if (len(moves) > 0):
            moves = np.concatenate(moves, axis = 1)
            moves = self.__removeOutOfBorderMoves(moves)
        
        return moves
        
    def _getBishopMoves (self, player_position, opponent_position, player, turn):
        
        # Current position
        wp = player_position[2] # Get the player position
        moves = self._getLongRangeMoves(self.bishop_move, wp ,player_position, opponent_position, player, turn)
        
        return moves
 
    def _getRookMoves (self,  player_position, opponent_position, player, turn):
        
        # Current position
        wp = player_position[3] # Get the player position
        moves = self._getLongRangeMoves(self.rook_move, wp ,player_position, opponent_position, player, turn)

        return moves
    
    def _getLadyMoves (self, player_position, opponent_position, player, turn):
        
        # Mix of Bishop and Rook, easy one
        wp = player_position[4] # Get the player position
        bishop_moves = self._getLongRangeMoves(self.bishop_move, wp ,player_position, opponent_position, player, turn)
        rook_moves = self._getLongRangeMoves(self.rook_move, wp ,player_position, opponent_position, player, turn)
        
        moves = []
        
        if (len(bishop_moves) != 0 and len(rook_moves) != 0):
            moves = np.concatenate([bishop_moves, rook_moves], axis = 1)
        elif (len(bishop_moves) != 0):
            moves = bishop_moves
        elif (len(rook_moves) != 0):
            moves = rook_moves
        
        return moves
    
    def _getKingMoves (self, player_position, opponent_position, player, turn):
        
        wp = player_position[5] # Get the player position
        moves = []
        
        if wp.shape[1] > 0:
            moves = np.concatenate([
                np.concatenate([wp for i in range(8)], axis = 1),
                self.king_move.T+wp
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
        
        last_row = 7 if player == 1 else 0
        
        old_board = board.copy()
        board[last_row, np.where(
            (np.mod(board[last_row,:], 10) == player) & \
            (board[last_row,:]) != 0
        )[0]] = (promotion*10)+player
        
        return board
    
    def _apply_move (self, move, board):
        
        """
            Basically apply virtually a move to a board without any sanity check
            
            Input :
                move to apply
            Ouput :
                board after appliance of the move
        """
        
        board_copy = board.copy()
        piece = board_copy[tuple(move[0])] 
        board_copy[tuple(move[0])] = 0 # Setting empty the previous position
        board_copy[tuple(move[1])] = piece # Setting the new position
        
        return board_copy
    
    def play_move(self, move, promotion = 5):
        
        """
            play_move :
                Play a move
                For that :
                    0. Play only if not mate
                    1. Check the validity of the move
                    2. Play the move
                    3. Promote pieces
                    4. Storing and recomputing the new state
                    5. Update the player and eventually the turn
                    6. Evaluate if there is a winner
                    
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
        new_board = self.board.copy()
        new_board = self._apply_move(move, new_board)
    
        # 3. Promote pieces
        new_board = self._promotion(new_board, self.current_player, promotion)

        # 4. Storing and recomputing the new state
        self.board = new_board
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
        if (self.is_check()):
            self.mate = self.is_mate()
        
        pass
    
    ###
    # Ensemble of functions for game evaluation
    ###
    
    def is_check(self):
        
        """
            Analyze is the current game is check
            
            Input : None
            Output : boolean, true if check, false otherwise
        """
        
        king_location = np.where(self.board == 60+self.current_player)
        if king_location[0].sum() > 0:
            king_location = np.array(king_location).reshape(2).tolist()
            check_situations = [True for x in self.opponentNextMoves if king_location == x[1]]
            is_check = len(check_situations) > 0
        else:
            is_check = False
        
        return is_check
    
    def is_mate(self):

        current_player_positions = self.__player_layers_positions[self.current_player]
        current_opponent_positions = self.__player_layers_positions[1-self.current_player]
        
        # Get the positions
        positions = np.array([np.where(x == 1) for x in self.board_layers_history[-1]], dtype = "object")
        player_position = [np.stack(x) for x in positions[current_player_positions]]
        opponent_position = [np.stack(x) for x in positions[current_opponent_positions]]

        king_moves = chess._getKingMoves(player_position, opponent_position, self.current_player, self.turn+1)

        mate_list = []
        if len(king_moves) > 0:
            
            king_moves = king_moves.T.reshape(king_moves.shape[1], 2, 2).tolist()

            for king_move in king_moves:
                
                king_move_board = self._apply_move(king_move, self.board) # Applying the move to the king
                king_move_next_move = self._getNextMove(
                    self._board_to_layer(king_move_board), 
                    1-chess.current_player, 
                    chess.turn + 1
                ) # Getting the next move of the opponents
                king_move_next_move = reduce(add, list(king_move_next_move.values())) # Getting the next move of the opponents
                
                # Is the king in the opponent next move ?
                king_move_mate = len([True for x in king_move_next_move if king_move[1] == x[1]]) > 0
                mate_list.append(king_move_mate)
        
        mate = True in mate_list
        
        return mate

# +
chess = playChess()

for i in range(1000):
    next_move = chess._getCurrentNextMove()
    if (len(next_move) > 0):
        new_board = chess.play_move(next_move[random.randint(0, len(next_move)-1)])
            
chess.board
