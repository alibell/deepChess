# -*- coding: utf-8 -*-
from deepChess import stockFish
from deepChess import chessBoard
from deepChess.chessBoard import directions_matrice_ref, moves_matrice_ref, moves_ref
from deepChess.players import kStockFishPlayer, deepChessPlayer

model = "./models/test.pt"

playerDeep = deepChessPlayer(player_id=0, model=model, keep_history=True)

playerDeep.next_move_prob(board)

player0 = kStockFishPlayer(player_id = 0, k = 0.2, keep_history = True, stockFish_path = "../stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")
player1 = kStockFishPlayer(player_id = 1, k = 0.8, keep_history = True, stockFish_path = "../stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2")

board = chessBoard.playChess()

# +
results = []

for j in range(100):
    print(f"Game n°{j+1}")
    board = chessBoard.playChess()

    player0.new_game()
    player1.new_game()
    
    players = [
        player0,
        player1
    ]
            
    while board.mate == False and board.is_draw()[0] == False:
        for j in [0,1]:
            # Play move
            move = players[j].play_move(board)
            if move is not None:
                #print(f"Player {j} : {str(move[0])} - Promotion : {move[1]}")
                pass
            # Check integrity
            sf_fen = players[j].get_fen_position(board)
            sf_fen_clean = " ".join(sf_fen.split(" ")[0:-3])
            local_fen = board.get_fen_position()
            local_fen_clean = " ".join(local_fen.split(" ")[0:-3])

            if sf_fen_clean != local_fen_clean:
                raise Exception("Discordance")

    if board.mate == True:
        results.append(board.winner)
    if board.is_draw()[0] == True:
        results.append("draw")
# -

import datetime
from deepChess.players import deepChessPlayer
from deepChess.chessBoard import playChess
from torch.utils.tensorboard import SummaryWriter


class MCTS ():
    
    """
        Tree of the Monte Carlo Tree Search
        The tree is composed by a succession of Nodes
        The nodes are created when a simulation phase start
        The transition between two nodes is declenched by a action.
        An action is a playing move, the opponent move goes straighforwards
    """
    
    def __init__ (player0, player1, model, device = "cpu", tensorboard_dir = "logs", log = False):
        
        """
            Initialization of the MCTS search
            The player0 is the MCTS player
            The player1 is its opponent
            
            Input :
                player0, player1:  instance of the player class object
                model : path of the model object, the model is used for the learned policy
                device : str, device in which we load the model (cpu or cuda)
                tensorboard_dir : str, folder in which we store the log of the current process
                log : boolean, true if we want the MCTS to log its activity in tensorboard
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
            self.mcts_uid = random.randint(0, 10e5)
            current_date = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%Y')
            self.mcts_tb_tag = str("/".join([current_date, str(self.nn_uid)]))
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
        
        self.players = {
            0:self.playerModel,
            1:self.player1
        }
        
    def play_game(self, n = 10, n_simulations = 100):
        
        """
            Play chess game according to MCTS rule
            
            Input :
                n : number of game to play
                n_simulations : number of simulation to perform
            Output :
                None
        """
        
        #
        # Play_game
        #
        
        """
            In this part, the MCTS will play n times against player1.
            For each game :
                0. Initialization of the game
                1. Selection : It will play according to the model policy, this is the selection phase
                2. Each loop will expand the tree, this is the expansion phase
                3. When a threshold it triggered, we enter in playout phase : in this phase, the policy is the player0 policy and n_simulations simulation are performed
                4. At the end of the game, the backpropagation is performed from the current batch size
        """
        
        for i in range(n):            
            # 0 : Initialization of the game
            
            ## 0.1 : Getting the players ready to play
            self.playerModel.new_game()
            self.player0.new_game()
            self.player1.new_game()
            self.players[0] = self.playerModel # The player is the default policy
            
            ## 0.2 : Getting the chess object
            chess = playChess()
            
            ## 0.3 : Recording the game in tensorboard
            if self.writer is not None:
                self.writer.add_text(
                    self.mcts_tb_tag+"/games",
                    f"Starting a new game - id : {i}"
                )
                
            ## 0.4 Error log
            n_errors = 0
            
            # 1/2/3. Playing the game
            
            ## Try catch, to prevent train break due to implementation error of chess game
            try:
                
                #playerDeep.next_move_prob(board)
            except:
                # Recording errors
                if self.writer is not None:
                    n_errors += 1
                    self.writer.add_text(
                        self.mcts_tb_tag+"/errors",
                        f"An error occurence in party - id : {i}"
                    )
            
            # 5. After the game
            
            ## 5.0 : Recording all the errors
            if self.writer is not None:
                self.writer.add_scalar(
                    self.mcts_tb_tag+"/errors_count",
                    n_errors
                )


class MCTS_Node ():
    """
        Node for Monte Carlo Tree Search
        Each node is composed by a state which is the chess object
        For each node, the leaf state is evaluated, a leaf nodes determine the end of the MCTS
        The node can initiate an action, diriged by the MCTS Tree, this action lead to a new node generation, until we reach a leaf node.
    """
