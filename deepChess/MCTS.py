# +
#
# MCTS.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class for processing the Monte Carlo Tree Search for next move calculation
#


import datetime
import random
import numpy as np
from functools import reduce
from operator import add
from deepChess.players import deepChessPlayer
from deepChess.chessBoard import playChess
import copy
from torch.utils.tensorboard import SummaryWriter

class MCTS ():
    
    """
        Monte Carlo Tree Search
        For a current game, the MCTS proceed to a number of simulation and pick among these the next move and next action
        The tree is composed by a succession of Nodes
        Each node select the best move according to the computation of UCB score to the default policy
        For this :
            Each state is identified by its fen representation
            Each action is identified by its hash
    """
    
    def __init__ (self, player0, player1, model, device = "cpu", tensorboard_dir = "logs", game_history_path = None, game_id = None, log = False):
        
        """
            Initialization of the MCTS search
            The player0 is the MCTS player, which play the moves following the first move
            The player1 is its opponent
            
            Input :
                player0, player1:  instance of the player class object
                model : path of the model object, the model is used for the learned policy
                device : str, device in which we load the model (cpu or cuda)
                tensorboard_dir : str, folder in which we store the log of the current process
                game_id : id of the current game for tensorboard monitoring, otherwise it will be generated
                log : boolean, true if we want the MCTS to log its activity in tensorboard
                game_history_path : path where to store the games for neural network training, they are stored in a pickle file
        """
        
        #
        # The tree can be composed of several branches
        # Each branch is composed of nodes is a dictionnary
        #
        
        branch = {
        }
        
        #
        # Loading tensorboard writter
        #
        
        if log == True and tensorboard_dir is not None:
            self.writer = SummaryWriter(tensorboard_dir)
            self.game_id = random.randint(0, 10e5)
            current_date = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%Y')
            self.mcts_tb_game_id = str("/".join([current_date, str(self.game_id)]))
        else:
            self.writer = None
        
        #
        # Loading the neural netword player
        #
        
        self.playerModel = deepChessPlayer(player_id=0, model=model, device = device, keep_history=False)
        
        
        #
        # Storing the players
        #
        
        self.player0 = player0
        self.player1 = player1

        #
        # Storing the state-actions parameters
        #

        self.sa_parameters = {
        }

        self.s_sum = {
        }

    def next_move(self, chess, n_simulations = 10):
        
        """
            Get the next move according to MCTS simulation
            
            Input :
                chess : chess game object
                n_simulations : number of simulation to perform
            Output :
                None
        """
        
        """
            In this part, the MCTS will pick a root move according to the neural network policy
            This, n_simulations moves will be generated
            For each simulation, the next move is picked according to the player0 policy.
        """

        if self.writer is not None:
            self.writer.add_text(
                self.mcts_tb_game_id+"/mtcs/run",
                f"Starting a MCTS move calculation"
            )

        # Getting the next moves
        next_move_list, next_move_promotion, next_move_prob = self.playerDeep.next_move_prob(chess)

        for i in range(n_simulations):

            # Copy of the game and getting the state fen representation
            chess_temp = chess.copy()

            # List of state actions for back propagation
            state_action = []
            
            # Computing state hash
            state = self._get_fen_position(chess_temp)

            # Synchronize player0 and player1 with the game
            self.player0.set_position(chess_temp.get_fen_position())
            self.player1.set_position(chess_temp.get_fen_position())
            
            # Playing the game
            # Playing the first move from deep nn policy
            while chess_temp.draw is None and chess_temp.mate is None:
                # Getting the UCB score
                actions = [self._get_action_hash(x) for x in next_move_list]
                next_move_list_score = self._ucb_next_moves(state, actions, next_move_prob)

                # Picking action
                move_id = np.argmax(next_move_list_score)
                action = actions[move_id]
                next_move = next_move_list[move_id]
                promotion = next_move_promotion[move_id]

                ## Updating the state-action and the state count sum
                self.sa_parameters[state][action]["n"] += 1
                self.s_sum[state] += 1

                # Player 0 plays
                chess_temp.play_move(next_move, promotion)
                self.player0.watch_game()

                # Player 1 plays
                self.player1.play_move(chess_temp)

                # Storing the state - action
                state_action.append((state, action))

                # Get next state and move list
                next_move_list, next_move_promotion, next_move_prob = self.player0.next_move_prob(chess)
                state = self._get_fen_position(chess_temp)

            # Back propagation
            if chess_temp.draw:
                reward = 0
            elif chess_temp.winner == 0:
                reward = 1
            else:
                reward = -1

            for sa in state_action:
                self.sa_parameters[state][action]["w"] += reward
                self.sa_parameters[state][action]["q"] = self.sa_parameters[state][action]["w"]/self.sa_parameters[state][action]["n"]
                

    def _ucb_next_moves (self, state, actions, probabilities):

        """
            Compute the ucb score for list of possible moves moves

            Input :
                state : hash of the current state
                actions : hash of the actions
                probabilities : list of actions probabilities
        """

        scores = [self._ucb_next_move(state,x,y) for x,y in zip(actions, probabilities)]

        return scores

    def _ucb_next_move (self, state, action, action_prob):

        """
            Compute the ucb score for curent moves

            Input :
                state : hash of the current state
                action : hash of the current action
                action_prop : current action probability
        """

        ns = self.s_sum[state]
        c = np.log((1+ns+self.c_base)/(self.c_base)) + self.c_init
        u = (c*action_prob*np.sqrt(ns))/(1+self.sa_parameters[state][action]["n"])

        score = self.sa_parameters[state][action]["q"]+u

        return score

    def _get_fen_position (self, chess):

        """
            Get the hash of a board for state value
            It also register the state in the parameters if it does not exist

            Input : board
        """

        fen_position = hash(chess.get_fen_position())

        if fen_position not in self.sa_parameters.keys():
            self.sa_parameters[fen_position] = {}
        if fen_position not in self.s_sum.keys():
            self.s_sum[fen_position] = 0

        return fen_position

    def _get_action_hash (self, action, state):

        """
            Get the hash of an action
            It also register the state in the parameters if it does not exist

            Input : 
                action : action numpy object
                state : state hash
        """

        action_hash = hash(action.tobytes())
        if action_hash not in self.sa_parameters[state].keys():
            self.sa_parameters[state][action_hash] = {
                "n":0,
                "w":0,
                "q":0
            }
        

        return action_hash