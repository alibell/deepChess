# -*- coding: utf-8 -*-
# +
#
# Evaluation.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Player and record the performance of the algorithm
#
# -

from deepChess import chessBoard
from deepChess.chessBoardFast import playChess
from deepChess.model import get_lastest_model, load, deepChessNN
from deepChess.players import kStockFishPlayer, deepChessPlayer
from deepChess.MCTS import MCTS
import datetime

# +
#
# Parameters
# -- Set parameters here
#
# -

log_folder = "/home/ali/deepChess/evaluations" # For game log
eval_name = "sf_10_mcts10"
model_folder = "/home/ali/deepChess/models"
device = "cpu"
stockFish_path = "../stockfish_14.1_linux_x64_avx2/stockfish_14.1_linux_x64_avx2"
shallowBlue_path = "./binaries/shallowblue"
opponent = kStockFishPlayer(1, k = 0.1, stockFish_path= stockFish_path)
opponent_mcts = kStockFishPlayer(1, k = 0.1, stockFish_path= stockFish_path)
n_mtcs = 10

# +
#
# Evaluation
# -- Playing and recording scores
#
# -

# Getting model path
model_path = get_lastest_model(model_folder, 2)[1]

# Neural network model
player0 = deepChessPlayer(0, device = device, model = model_path)

# Loading mcts
mcts = MCTS(player0, opponent_mcts, model_path, device = device, tensorboard_dir=None, game_history_path=None, game_id = None, log = False)

# Output name : name of the file in wich we record the game
output_name = f"{log_folder}/{eval_name}.csv"

i = 0
while True:
    # Starting new play
    i += 1
    print("Starting new play")

    # Get the lastest model
    new_model_path = get_lastest_model(model_folder, 1)[0]
    if new_model_path != model_path:
        model_path = new_model_path
        mcts.update_model(model_path)
        player0.update_model(model_path)

    try:
        chess = playChess(shallowBlue_path)
        player0.new_game()
        opponent.new_game()
        opponent_mcts.new_game()

        while (((chess.draw is None) or (chess.draw[0] == False)) and (chess.winner is None)):
            if chess.current_player == 0:
                move = mcts.next_move(chess, n_mtcs)
                chess.play_move(move[0])
            elif chess.current_player == 1:
                opponent.play_move(chess)

        # Recording game
        winner = chess.winner
        draw = chess.draw[0]
        date = datetime.datetime.strftime(datetime.datetime.now(), "%m-%d-%Y - %H:%m:%S")
        model_file = model_path.split("/")[-1]
        print(f"End game - Winner : {winner} - Draw : {draw}")
        
        with open(output_name,"a") as file:
            file.write(f"{i},{date},{model_file},{winner},{draw}")
            file.write("\n")

    except Exception as e:
        print(f"Error occurence, game n°{i} aborded")
        print(f"Error code : {e}")
        pass
