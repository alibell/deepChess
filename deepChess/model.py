# +
#
# model.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class for estimation of the policy and the value function
# These are modelized threw a deep neural network with the pytorch library
#
# -

from torch.nn import Conv2d, Module, ReLU, MaxPool2d, Flatten, Linear, Softmax, Sequential, utils
from torch.optim import Adam
from torch import dtype
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import datetime

class rl_loss (Module):
    
    """
        Custom loss for RL training
    """
    
    def __init__ (self, regularization = 0.01):
        
        """
            Input : regularization parameter
        """
        super(rl_loss, self).__init__()
        
        self.regularization = regularization
        
    def forward (self, y_true, y_hat, parameters):
        
        """
            Input :
                x : tuple containing the board matrice x[0] and the features x[0]
                y : tuple containing the current value function and the MTCS policy
                parameters : coefficients of the neural network
                
            Output : loss value
        """
        
        true_value, mcts_policy = y_true
        estimed_value, estimed_policy = y_hat
        
        rl_loss = torch.square(true_value-estimed_value) \
                - torch.matmul(mcts_policy, estimed_policy.T) \
                + self.regularization*torch.square(utils.parameters_to_vector(parameters)).sum()
        
        rl_loss = rl_loss/true_value.shape[0]
        
        return rl_loss
    
class deepChessNN (Module):
    
    def __init__ (self, channels = 96, regularization = 0.01, lr = 0.001, betas = (0.9, 0.999), weight_decay = 0, tensorboard_dir = "./logs"):
        
        """
            Initialization of the neural network
            Input :
                channels : number of channel of the input matrice
                regularization : regularization parameter for the loss function
                lr : learning rate
                betas : betas value for Adam optimizer
                weight_decay : weight_decay value for the optimizer
        """
        
        super(deepChessNN, self).__init__()
        
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
        
        self._convolutions_operations = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in convolution_parameters]
        
        self.convolutions_operations = Sequential(*self._convolutions_operations)
        
        linear_parameters = [
            {
                "module": Linear,
                "in_features": 1610,
                "out_features": 800,
                "bias":True
            },
            {
                "module": ReLU
            }
        ]
        
        self._linear_operations = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in linear_parameters]
        self.linear_operations = Sequential(*self._linear_operations)
        
        value_projection_parameters = [
            {
                "module": Linear,
                "in_features": 800,
                "out_features": 400,
                "bias":True
            },
            {
                "module": ReLU
            },
            {
                "module": Linear,
                "in_features": 400,
                "out_features": 100,
                "bias":True
            },
            {
                "module": ReLU
            },
            {
                "module": Linear,
                "in_features": 100,
                "out_features": 25,
                "bias":True
            },
            {
                "module": ReLU
            },
            {
                "module": Linear,
                "in_features": 25,
                "out_features": 1
            }
        ]
        
        self._value_projection_operations = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in value_projection_parameters]
        self.value_projection_operations = Sequential(*self._value_projection_operations)
        
        moves_projection_parameters = [
            {
                "module": Linear,
                "in_features": 800,
                "out_features": 2000,
                "bias":True
            },
            {
                "module": ReLU
            },
            {
                "module": Linear,
                "in_features": 2000,
                "out_features": 4672,
                "bias":True
            },
            {
                "module": Softmax,
                "dim":1
            }
        ]
        
        self._moves_projection_operations = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in moves_projection_parameters]
        self.moves_projection_operations = Sequential(*self._moves_projection_operations)
        
        # Initiation of the loss
        
        self.loss = rl_loss(regularization)
        
        # Initiation of the optimizer
        self.optimizer = Adam(self.parameters(), lr = lr, betas = betas, weight_decay=weight_decay)
        
        # Losses
        self.losses = []
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(tensorboard_dir)
        self.nn_uid = random.randint(0, 10e5)
        current_date = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%Y')
        self.nn_tb_tag = str("/".join([current_date, str(self.nn_uid)]))
    
    def _register_loss(self, loss):
        
        """
            Function that register the loss value
            
            Input :
                loss : scalar value of the loss
            Output : None
        """
        
        # Register the loss
        self.losses.append(loss)
        
        # Send the loss to tensorboard
        self.writer.add_scalar(
            "/".join([self.nn_tb_tag, "loss"]),
            loss
        )
        
        return None
    
    def forward (self, x):
        
        """
            Forward pass
            
            Input :
                x : tuple containing the board matrice x[0] and the features x[0]
            Output :
                tuple containing the value function and the next move probability distribution
        """

        # 1. Encoding
        board = x[0]
        features = x[1] 

        ## 1.1. Encoding with CNN
        board = self.convolutions_operations(board)
            
        ## 1.2. Encoding for features
        board_features = torch.concat([board, features], axis = 1)    
        
        ## 1.3. Projection of all together
        board_features = self.linear_operations(board_features)
        
        # 2. Decoding
        
        ## 2.1 Value function
        value_function = board_features.clone()
        value_function = self.value_projection_operations(value_function)        
            
        ## 2.2 Probability distribution
        moves = board_features
        moves = self.moves_projection_operations(moves)            

        return value_function, moves
    
    def train (self, x, y):
        
        """
            Forward pass
            
            Input :
                x : tuple containing the board matrice x[0] and the features x[0]
                y : tuple containing the current value function and the MTCS policy
            Output :
                tuple containing the value function and the next move probability distribution
        """
        
        # Zero grad
        self.optimizer.zero_grad()
        
        # Forward pass
        y_hat = self.forward(x)
        
        # Loss computation
        loss = self.loss(y_hat, y, self.parameters())
        
        # Back propagation
        loss.backward
        
        # Gradient descent
        self.optimizer.step()
        
    def predict(x):
        
        # Prediction
        
        with torch.no_grad():
            y_hat = self.forward(x)
            
        return y_hat