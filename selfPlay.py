# -*- coding: utf-8 -*-
# +
#
# selfPlay.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Python script that generate games over selfplay
#
# -

from deepChess import chessBoard
from deepChess.chessBoardFast import playChess
from deepChess.model import get_lastest_model, load, deepChessNN
from deepChess.players import kStockFishPlayer, deepChessPlayer
from deepChess.MCTS import MCTS
import random
import pickle
import os

# +
#
# Parameters
# -- Set parameters here
#
# -

data_folder = "/home/ali/deepChess/games"
model_folder = "/home/ali/deepChess/models"
log_folder = "/home/ali/deepChess/logs"
device = "cuda:0"
shallowBlue_path = "./binaries/shallowblue"
stockFish_path = "../stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2"
players = {
    0:{
        'k': 0.8, 
        'keep_history': False
    },
    1:{
        'k': 0.9, 
        'keep_history':False
    }
}
n_mtcs = 200
n_games = 100

# +
#
# Self play
# -- Playing and recording games
#

# +
# Getting players
current_players = {}

for x,y in players.items():
    current_players[x] = kStockFishPlayer(player_id=x, stockFish_path=stockFish_path, **y)
current_players[11] = kStockFishPlayer(player_id=1, stockFish_path=stockFish_path, **players[1])
# -

# Getting model path
model_path = get_lastest_model(model_folder, 5)[4] # Not taking the last one, because it can be in writting

# Generate run uid
uid = random.randint(0, 10e6)

# Loading mcts
mcts = MCTS(current_players[0], current_players[1], model_path, device = device, tensorboard_dir=log_folder, game_history_path=data_folder, game_id = str(uid), log = True)

for i in range(n_games):
    # Get the lastest model
    new_model_path = get_lastest_model(model_folder, 5)[4]
    if new_model_path != model_path:
        model_path = new_model_path
        mcts.update_model(model_path)
    
    try:
        print(f"Game n°{i}")
        chess = playChess(shallowBlue_path)
        current_players[0].new_game()
        current_players[1].new_game()
        current_players[11].new_game()

        game_history = [] # History of the moves
        
        while (((chess.draw is None) or (chess.draw[0] == False)) and (chess.winner is None)):
            if chess.current_player == 0:
                move = mcts.next_move(chess, n_mtcs)
                chess.play_move(move[0])

                # Store MCTS history
                game_history.append(mcts._get_game_history(chess))
            elif chess.current_player == 1:
                current_players[11].play_move(chess)

        # Get the score
        if chess.winner is not None:
            score = 1 if chess.winner == 0 else -1
        else:
            score = 0      
        
        print(chess.winner)
        print(f"End of the game n°{i} - Score : {score}")

        # Update MCTS history
        for j in range(len(game_history)):
            game_history[j] = (*game_history[j], score)
        
        # Saving history for NN training
        if data_folder is not None:
            # File path
            file_folder = "/".join([data_folder, mcts.mcts_tb_game_id])
            file_path = f"{file_folder}/{i}.pickle"

            # Creating dir
            if not os.path.exists(file_folder):
                os.makedirs(file_folder)

            with open(file_path, 'wb') as file:
                pickle.dump(game_history, file)

                
    except Exception as e:
        print(f"Error occurence, game n°{i} aborded")
        print(f"Error code : {e}")
        pass


