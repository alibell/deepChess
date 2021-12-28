# -*- coding: utf-8 -*-
# +
#
# api.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Python script that serve an HTTP API for the algorithm play
# Allow to create a web interface to play against the algorithm
#
# -

import random
from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
app.config["DEBUG"] = False
CORS(app)

from deepChess.shallowBlue import shallowBlue_connector
from deepChess.chessBoardFast import playChess
from deepChess.MCTS import MCTS
from deepChess.players import deepChessPlayer
from deepChess.model import get_lastest_model

# List of current games : store the game object of each one
games = {}

# Token
auth_token = "IDb6IHMzrcG25MnRnFWAqt4HDNAUTJ2OIV42WdWtSLtkd7IwVm"

# Launch an empty board which will be copied as necessary
sb_path = "./binaries/shallowblue"
model_folder = "/home/ali/deepChess/models"
device = "cuda:0"
n_mtcs = 15
max_turn = 50

chess = playChess(sb_path)

# Launching deepChess player and MCTS
model_path = get_lastest_model(model_folder, 2)[1]

player0 = deepChessPlayer(1, device = device, model = model_path)
mcts = MCTS(player0, player0, model_path, device = device, tensorboard_dir=None, game_history_path=None, game_id = None, log = False, mcts_player = 1, max_turn = max_turn)

@app.route('/start/<token>')
def start_game(token):

    # No token, no game
    if token == auth_token:
        # Generate UID
        uid = str(random.randint(0, 10e6))

        # Launch the game
        games[uid] = chess.copy()
        current_player = games[uid].current_player

        print(f"New Game, uid = {uid}, first player : {current_player}")

        # Return uid
        return uid

@app.route("/play/<uid>/<move>")
def play_move(uid, move):
    try:
        if games[uid].current_player == 0:
            move = games[uid]._stockFishToMove(move, 1-games[uid].first_player)
            games[uid].play_move(move)

            return games[uid].get_fen_position()
        else:
            return games[uid].get_fen_position()
    except:
        return games[uid].get_fen_position()

@app.route("/play/<uid>/opponent")
def play_opponent(uid):
    move = mcts.next_move(games[uid], n_mtcs)
    games[uid].play_move(move[0])

    return games[uid].get_fen_position()

@app.route("/play/is_mate")
def is_mate(uid):
    is_mate = 1 if games[uid].mate else 0

    return is_mate

app.run(host = "192.168.1.93", debug = False, port = 5000)
