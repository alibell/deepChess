# -*- coding: utf-8 -*-
# +
#
# train.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Python script for neural network training
#
# -

from deepChess.model import get_lastest_model, load
from deepChess.chessBoard import playChess
import os
import torch
import glob
import pickle
import time

# +
#
# Parameters
# -- Set parameters here
#
# -

data_folder = "/home/ali/deepChess/games"
model_folder = "/home/ali/deepChess/models"
log_folder = "/home/ali/deepChess/logs"
lr = 0.01 # Learning rate of the training
device = "cuda:0" # Device in which the model is loaded
timer = 60*1 # Timer in seconds before attempting again to train the network
max_epoch = 100 # Number of time it can has seen the same data
max_model = 100 # Maximum number of model record to keep
batch_size = 256 # Batch size

# +
#
# Train
# -- Training the model
#
# -

# Getting the model to train
model_path = get_lastest_model(model_folder, 2)[1]
model = load(model_path)
model = model.to(device)

# Setting the learning rate
model.optimizer.param_groups[0]['lr'] = lr

# Getting train history
train_history_path = data_folder+"/history"

if os.path.exists(train_history_path):
    history = pickle.load(open(train_history_path, "rb"))
else:
    history = {}

# +
# For each, we load the data and train the NN from it
# -

while True:
    # Getting the data for NN training
    games = glob.glob(data_folder+"/**/*.pickle", recursive=True)
    games = [x for x in games if (x in history.keys() and history[x] <= max_epoch) or (x not in history.keys())]

    n_batch = len(games)//batch_size+int((len(games)%batch_size) != 0)

    for batch in range(n_batch):
        batch_games = games[batch*batch_size:(batch+1)*batch_size]

        # Only keeping the last models
        last_models = get_lastest_model(model_folder, max_model)
        for model_to_delete in [x for x in glob.glob(model_folder+"/*.pt") if x not in last_models]:
            os.remove(model_to_delete)

        print(f"Passing batch - Batch size {batch_size}")

        # Loading data
        batch_tensor = {
            "X":[],
            "y":[]
        }

        for game in batch_games:

            # Registering the game
            if game not in history.keys():
                history[game] = 0

            game_data = pickle.load(open(game, "br"))
            X = tuple(torch.tensor(x,dtype = torch.float32, device = device) for x in game_data[1])
            moves = [torch.tensor(
                        playChess._localToNNMove(None, [x[0]], game_data[0], 0)[1], dtype = torch.float32, device = device).flatten()
                                for x in game_data[2]]
            rewards = [torch.tensor(x[1], dtype = torch.float32, device = device) for x in game_data[2]]

            if len(moves) > 0:
                X = (
                    torch.stack([X[0] for x in range(len(moves))]),
                    torch.stack([X[1] for x in range(len(moves))])
                )
                batch_tensor["X"].append(X)
                y = (torch.stack(rewards), torch.stack(moves))
                batch_tensor["y"].append(y)

            # Saving the fit in history
            history[game] += 1

        # Fitting the model
        if len(batch_tensor["X"]) > 0:
            print("Training model")

            X = (torch.concat([x[0] for x in batch_tensor["X"]]), torch.concat([x[1] for x in batch_tensor["X"]]))
            y = (torch.concat([x[0] for x in batch_tensor["y"]]), torch.concat([x[1] for x in batch_tensor["y"]]))

            model.fit(X, y)

        # Saving the new model
        next_model_path = "/".join([
            model_folder,
            str(int(".".join(model_path.split("/")[-1].split(".")[0:-1]))+1)
        ])+".pt"
        model_path = next_model_path
        model.save(model_path)

        # Saving the history
        with open(train_history_path, "wb") as file:
            pickle.dump(history, file)
    
    time.sleep(timer)
