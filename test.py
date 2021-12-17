from deepChess import stockFish
from deepChess import chessBoard

sf = stockFish.stockFish_connector("../stockfish/stockfish_14.1_win_x64_avx2/stockfish_14.1_win_x64_avx2.exe")

chess = chessBoard.playChess()

from torch.nn import Conv2d, Module, ReLU, MaxPool2d, Flatten, Linear
from torch import dtype
import torch


class DeepChessNN (Module):
    def __init__ (self, channels = 96):
        super(DeepChessNN, self).__init__()
        
        convolution_parameters = [
            {
                "module": Conv2d,
                "in_channels": channels,
                "out_channels": 100,
                "kernel_size": (4,4),
                "bias":True,
                "padding":'same'
            },
            {
                "module": ReLU
            },
            {
                "module": MaxPool2d,
                "kernel_size": (2,2)
            },
            {
                "module": Conv2d,
                "in_channels": 100,
                "out_channels": 200,
                "kernel_size": (5,5),
                "bias":True,
                "padding":'same'
            },
            {
                "module": ReLU
            },
            {
                "module": MaxPool2d,
                "kernel_size": (2,2)
            },
            {
                "module": Conv2d,
                "in_channels": 200,
                "out_channels": 400,
                "kernel_size": (1,1),
                "bias":True,
                "padding":'same'
            },
            {
                "module": Flatten
            }
        ]
        
        self.convolutions_operations = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in convolution_parameters]
        
    def forward (self, x):

        # 1. Encoding
        board = x[0]
        features = x[1] 

        ## 1.1. Encoding with CNN
        for i in range(len(self.convolutions_operations)):
            board = self.convolutions_operations[i](board)
            
        ## 1.2. Encoding for features
        board_features = torch.concat([board, features], axis = 1)    
        
        ## 1.3. Projection of all together
        
        # 2. Decoding

        return board


deepChess = DeepChessNN()

test = torch.tensor(np.stack([chess.current_board_to_NN_input()[0]]), dtype = torch.float32)
test2 = torch.tensor(np.stack(chess.current_board_to_NN_input()[1]), dtype = torch.float32).reshape(1,10)

a = deepChess.forward((test,test2))
