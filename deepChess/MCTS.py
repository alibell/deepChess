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
import pickle
import os

from numpy.core.fromnumeric import argmax
from deepChess.players import deepChessPlayer
from deepChess.chessBoard import playChess
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
    
    def __init__ (self, player0, player1, model, device = "cpu", tensorboard_dir = "logs", game_history_path = None, game_id = None, log = False, max_turn = 30, mcts_player = 0, dir_noise = 0.3, exploration_prop = 0.25):
        
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
                max_turn : limit the number of turn in an MCTS search, otherwise it is too long on my poor computer
                mcts_player : player of the MCTS, by default 0
                dir_noise : noise to add to the prior probability
                exploration_prop : proportion represented by the noise in the UCB computation
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
            if game_id is None:
                self.game_id = random.randint(0, 10e5)
            else:
                self.game_id = game_id
            current_date = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%Y')
            self.mcts_tb_game_id = str("/".join([current_date, str(self.game_id)]))
        else:
            self.writer = None
        
        #
        # Loading the neural network player
        #
        self.device = device
        self.playerModel = deepChessPlayer(player_id=mcts_player, model=model, device = device, keep_history=False)
        
        
        #
        # Storing the players
        #
        
        self.player0 = player0
        self.player1 = player1
        self.mcts_player = mcts_player
        

        #
        # Storing the state-actions parameters
        #

        self.sa_parameters = {
        }

        self.s_sum = {
        }

        #
        # Global parameters
        #

        #??c_base and c_init : weird but given by the authors, not much documentation about it
        self.game_history_path = game_history_path
        self._pickle_count = 1
        self.c_base = 19652
        self.c_init = 1.25
        self.dir_noise = dir_noise
        self.exploration_prop = exploration_prop
        self.max_turn = max_turn
        self.state_init = ''
        
        #
        # Actions memory
        #

        self._actions = []

    def update_model(self, model_path):
        """
            Update the neural network model

            Input :
                model_path
        """

        self.playerModel = deepChessPlayer(player_id=0, model=model_path, device = self.device, keep_history=False)

    def next_move(self, chess, n_simulations = 100):
        
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

        #??Not valid if the current player is not player 0
        if chess.current_player != self.mcts_player:
            return None

        if self.writer is not None:
            self.writer.add_text(
                self.mcts_tb_game_id+"/mtcs/run",
                f"Starting a MCTS move calculation"
            )

        # Reset the current search
        self.sa_parameters = {
        }

        self.s_sum = {
        }
        
        # Reset the initial state
        self.state_init = self._get_fen_position(chess)

        # Getting the next moves
        next_move_list_init, next_move_promotion_init, next_move_prob_init = self.playerModel.next_move_prob(chess)
        next_move_list, next_move_promotion, next_move_prob = next_move_list_init, next_move_promotion_init, next_move_prob_init 

        for i in range(n_simulations):

            # Copy of the game and getting the state fen representation
            chess_temp = chess.copy()

            # List of state actions for back propagation
            state_action = []
            # Synchronize player0 and player1 with the game
            self.player0.set_position(chess_temp.get_fen_position())
            self.player1.set_position(chess_temp.get_fen_position())
            
            # Playing the game
            # Playing the first move from deep nn policy
            while ((chess_temp.draw is None) or (chess_temp.draw[0] == False)) and (chess_temp.winner is None) and (chess_temp.turn <= self.max_turn):
                try:
                    #??Computing state hash
                    state = self._get_fen_position(chess_temp)

                    # Getting the UCB score
                    actions = [self._get_action_hash(x, state) for x in next_move_list]
                    next_move_list_score = self._ucb_next_moves(state, actions, next_move_prob, dir_noise = self.dir_noise)

                    # Picking action
                    if len(next_move_list_score) > 0:
                        move_id = np.argmax(next_move_list_score)
                    else:
                        break
                    action = actions[move_id]
                    next_move = next_move_list[move_id]
                    promotion = next_move_promotion[move_id]

                    ## Updating the state-action and the state count sum
                    self.sa_parameters[state][action]["n"] += 1
                    self.s_sum[state] += 1

                    # Player 0 plays
                    chess_temp.play_move(next_move, promotion)
                    self.player0.watch_game(chess_temp)

                    # Player 1 plays
                    self.player1.play_move(chess_temp)
                    self.player0.watch_game(chess_temp)

                    # Storing the state - action
                    state_action.append((state, action))

                    # Get next move list
                    next_move_list, next_move_promotion, next_move_prob = self.player0.next_move_prob(chess_temp)

                except Exception as e:
                    # Break loop if there is a game error
                    print(f"MCTS Exception : {e}")
                    if self.writer is not None:
                        self.writer.add_text(
                            self.mcts_tb_game_id+"/mtcs/run",
                            f"Error in game : break loop"
                        )
                    break

            # Back propagation
            if chess_temp.draw[0]:
                reward = 0
            elif chess_temp.winner == 0:
                reward = 1
            else:
                reward = -1

            for sa in state_action:
                self.sa_parameters[sa[0]][sa[1]]["w"] += reward
                if self.sa_parameters[sa[0]][sa[1]]["n"] > 0:
                    self.sa_parameters[sa[0]][sa[1]]["q"] = self.sa_parameters[sa[0]][sa[1]]["w"]/self.sa_parameters[sa[0]][sa[1]]["n"]
                else:
                    self.sa_parameters[sa[0]][sa[1]]["q"] = 0
            
            # Reinitialisation of the moves
            next_move_list, next_move_promotion, next_move_prob = next_move_list_init, next_move_promotion_init, next_move_prob_init 

        # Getting the best move
        values = np.array([x["q"] for x in self.sa_parameters[self.state_init].values()])
        if len(values) > 0:
            best_move_id = np.argmax(values)
            best_move = self._actions[list(self.sa_parameters[self.state_init].keys())[best_move_id]]
            reward = values[best_move_id]
        else:
            n_next_moves = len(next_move_list_init)
            best_move_id = random.randint(0, n_next_moves-1)
            best_move = next_move_list_init[best_move_id]
            reward = 0

        # Storing game history
        #if len(values) > 0:
            #if self.game_history_path is not None:
                #self._save_game_history(self.state_init, chess)

        if self.writer is not None:
            self.writer.add_text(
                self.mcts_tb_game_id+"/mtcs/run",
                f"Ending a MCTS move calculation - Move : {best_move} - Reward = {reward}"
            )

        return best_move, 

    def _ucb_next_moves (self, state, actions, probabilities, dir_noise = None):

        """
            Compute the ucb score for list of possible moves moves

            Input :
                state : hash of the current state
                actions : hash of the actions
                probabilities : list of actions probabilities
                dir_noise : value of the parameter of dirichlet noise to add to the vector, if None it is ignored
        """

        probabilities = np.array(probabilities)
        if dir_noise is not None:
            noise = np.random.dirichlet(dir_noise*np.ones(len(probabilities)))

            probabilities = (1-self.exploration_prop)*probabilities+self.exploration_prop*noise

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

        if action not in self._actions:
            self._actions.append(action)

        action_hash = self._actions.index(action)

        if action_hash not in self.sa_parameters[state].keys():
            self.sa_parameters[state][action_hash] = {
                "n":0,
                "w":0,
                "q":0
            }
        
        return action_hash

    def _get_game_history (self, chess):

        """
            Get the game history after a move seach

            Input : chess board
            Output : tuple of values
        """

        # Get vector representation of the chess board
        board_input = chess.current_board_to_NN_input()
        board_matrix = chess.board

        # Get MCTS values - policies pair
        policies = [[self._actions[x[0]], x[1]["q"]] for x in self.sa_parameters[self.state_init].items()]
        policies_q = self.__softmax(np.array([x[1] for x in policies])).tolist()
        
        for i in range(len(policies)):
            policies[i][1] = policies_q[i]

        history = (board_matrix, board_input, policies)

        return history

    def __softmax (self, array):

        """
            Apply the softmax function
        """

        res = np.exp(array)/np.exp(array).sum()
        return res

    def _save_game_history (self, state_init, chess):

        """
            Save the game history to a pickle file

            Input :
                state_init : hash of the initial state
                chess : object of the chess game
        """

        # Get vector representation of the chess board
        board_input = chess.current_board_to_NN_input()
        board_matrix = chess.board

        # Get MCTS values - policies pair
        policies = [[self._actions[x[0]], x[1]["q"]] for x in self.sa_parameters[state_init].items()]
        policies_q = [x[1] for x in policies]
        policies = [x for x in policies if x[1] == max(policies_q)]

        history = (board_matrix, board_input, policies)

        # Saving the pickle file
        if self.game_history_path is not None:
            # File path
            file_folder = "/".join([self.game_history_path, self.mcts_tb_game_id])
            file_path = f"{file_folder}/{self._pickle_count}.pickle"

            #??Creating dir
            if not os.path.exists(file_folder):
                os.makedirs(file_folder)

            with open(file_path, 'wb') as file:
                pickle.dump(history, file)
                self._pickle_count += 1

                if self.writer is not None:
                    self.writer.add_text(
                        self.mcts_tb_game_id+"/mtcs/run",
                        f"Game history saved - {file_path}"
                    )