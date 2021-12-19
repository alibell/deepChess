# -*- coding: utf-8 -*-
# +
#
# players.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class of chess players, these are python object which can analyse a chess board and pick a move
#
# -

import random

if __name__ == '__main__':
    from errors import raiseException
    from stockFish import stockFish_connector
else:
    from .errors import raiseException
    from .stockFish import stockFish_connector


class Player():
    """
        Player class
        Class from which will inherite all of the players subclass
    """
    def __init__(self, player_id, keep_history = False):
        
        if player_id in [0,1]:
            self.player = player_id
        else:
            raiseException("unknown_player")
        
        self.keep_history = keep_history
        self.history = {}
        self.history["moves"] = []
            
    def play_move(self, chess):
        
        """
            Playing a move : this function call the playing move function from the chess game
            Input : object of the current chess game
        """
        
        if chess.current_player == self.player:
            
            # Pre processing
            self._play_move_preprocessing(chess)
            
            # Getting the move
            next_move = self.next_move(chess)
            
            # playing the move
            chess.play_move(next_move[0], next_move[1])

            if self.keep_history:
                self.history["moves"].append((next_move[0], next_move[1]))
            
            # Post processing
            self._play_move_postprocessing(chess, next_move)
            
            return next_move
            
        else:
            pass

    def new_game(self):

        """
            Modify internal parameters for a new game
        """

        self._new_game()

        pass
    
    def _play_move_postprocessing(self, chess, next_move):
        
        """
            Function for postprocessing after playing a move
        """
            
    def _play_move_preprocessing(self, chess):
        
        """
            Function for preprocessing before playing a move
        """
        
        pass

    def _new_game(self):
        """
            Custom function for new game configuration
        """

class kStockFishPlayer(Player):
    
    """
        kStockFish :
            Player that play either given StockFish policy of given a random policy.
            The k parameter guides the proportion of Stockfish moves that the player will do.
            Thus with k = 0, the player plays totally randomly, with k = 1, it plays totally like stockfish
    """
    
    def __init__ (self, player_id, k = 0.5, stockFish_path = None, keep_history = False):
        
        """
            Initialization of the class
            Input :
                player_id : integer, 0 or 1, id of the player
                k : float, between 0 and 1, proportion of stockfish moves to play
                keep_history : it True, an history of the moves will be recorded
        """
        super(kStockFishPlayer, self).__init__(player_id, keep_history)
        
        # Check the k
        if k >= 0 and k <= 1:
            self.k = k
        else:
            raiseException("invalid_k_player")
            
        if stockFish_path is None:
            self.k = 0
            raise Warning("No stockfish executable was providen. The player will play 100% random (k = 0).")
         
        # Loading stockFish
        if self.k != 0:
            self.sf = stockFish_connector(stockFish_path)
        
        # Defining policies
        self.policies = {
            0:self._randomPolicy,
            1:self._sfPolicy
        }
        
        # Promotions dictionnary
        self.__promotions = {
            "q":5,
            "r":4,
            "b":3,
            "n":2
        }

        # History
        if self.keep_history:
            self.history["sf_moves"] = [] # Generated sf moves, None when random policy
            self.history["sf_replicates"] = [] # Replicated sf moves
            
    def _selectPolicy(self):
        
        """
            Select the policy with the probability k
        """
        
        if self.k in [0,1]:
            return self.k
        else:
            return int(random.random() < self.k)
    
    """
        Getting the next move
    """
    
    def next_move(self, chess):
        
        """
            This function compute the next move according to the player policy.
            The move should be formatted for the chess chess.
            
                input :
                    chess : object of the current chess game
                output :
                    next_move : tuple with :
                        list of origin and destination coordonates
                        promotion
        """
        
        # Getting the policy
        policy_id = self._selectPolicy()
        policy = self.policies[policy_id]
        
        # Getting the next move
        next_move = policy(chess)
        promotion = 5

        # If stockFish, we should convert the next move
        if next_move is not None:
            if policy_id == 1:
                if next_move["promotion"] is not None:
                    promotion = self.__promotions[next_move["promotion"]]
                next_move = chess._stockFishToMove(next_move["move"], first_move=1-chess.first_player)
        
        return (next_move, promotion)
    
    """
        Playing the move in the stockfish board
    """
    
    def _play_move_postprocessing(self, chess, next_move):
        
        """
            Function that is executed after playing move
            It post-process by playing the move in the stockfish board
            
            Input :
                chess : chess game object
                next_move : applied move
        """
        
        if self.k > 0:
            self._play_move_preprocessing(chess, process_current_player = True)
            
    def _play_move_preprocessing(self, chess, process_current_player = False):
        
        """
            Function that is executed before playing a move
            It pre-process by playing the opponent move in the board
            
            Input :
                chess : chess game object
                process_current_player : if True, the function with play the previous move event if the current player target the function
        """
        if self.k > 0:
            # Getting the move
            if chess.previousMove is not None:
                if (chess.previousMove[0] == (1-self.player)) or (process_current_player == True): # If the opponent played the previous move
                    opponent_move = chess.previousMove[1]
                    piece = chess.previousMove[2]

                    # Conversion of the move
                    sf_move = chess._movesToStockFish([opponent_move], 1-chess.first_player)[0] # If playing an history of move

                    # Adding the promotion if there is one
                    sf_promotion = ''
                    if (piece // 10) == 1: # Reconstite if there was a promotion
                        new_piece = chess.board[tuple(opponent_move[1])] // 10
                        if new_piece != 1:
                            sf_promotion = list(self.__promotions.keys())[list(self.__promotions.values()).index(new_piece)]

                    sf_move += sf_promotion

                    if self.keep_history:
                        self.history["sf_replicates"].append(sf_move)

                    # Playing the move in the sf board
                    self.sf.sf.make_moves_from_current_position([sf_move])

    """
        Defining the policies
    """
    
    def _randomPolicy(self, chess):
        
        """
            _randomPolicy : pick a move randomly among all possible moves
                Input : current chess game
                Output : the next move, None if it is not the player turn
        """
        
        if chess.current_player == self.player:
            
            next_moves = chess.getCurrentNextMove()

            n_moves = len(next_moves)
            move = random.randint(0, n_moves-1)
            
            next_move = next_moves[move]
        else:
            next_move = None

        if self.keep_history:
            self.history["sf_moves"].append(None)

        return next_move
    
    def _sfPolicy(self, chess):
        
        """
            _sfPolicy : pick a move according to SF policy
                Input : current chess game
                Output : the next move, None if it is not the player turn
        """
        
        # Checking that the chess signature is the same than SF one
        # @TODO, generate sf signature
        
        if chess.current_player == self.player:
            
            next_moves = self.sf.get_top_moves(5) # Picking one of the top 5 moves of SF
            move = random.randint(0, (len(next_moves)-1))
            
            next_move = next_moves[move]
        else:
            next_move = None

        if self.keep_history:
            self.history["sf_moves"].append(next_move)       

        return next_move

    """
        New game : resetting the player to initial states
    """

    def _new_game (self):

        """
            Custom new_game function, if stockFish : it will reset the sf state
        """

        if self.k > 0:
            self.sf.sf.set_position()

