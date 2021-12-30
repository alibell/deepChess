# +
#
# model.py
# A. Bellamine, L-D. Azoulay, N. Berrebi
# Class for estimation of the policy and the value function
# These are modelized threw a deep neural network with the pytorch library
#
# -

from torch.nn import Conv2d, Module, ReLU, MaxPool2d, Flatten, Linear, Softmax, Sequential, utils, BatchNorm2d, Tanh, ModuleList
from torch.optim import Adam
from torch import dtype
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import datetime
import glob

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
        
    def forward (self, y_hat, y_true, parameters):
        
        """
            Input :
                y_true : tuple containing the current value function and the MTCS policy
                y_hat : tuple containing the estimated value function and the estimated policy
                parameters : coefficients of the neural network
                
            Output : loss value
        """
        
        true_value, mcts_policy = y_true
        estimed_value, estimed_policy = y_hat
        estimed_value = estimed_value.reshape(estimed_value.shape[0])

        rl_loss = torch.square(true_value-estimed_value) \
                - (mcts_policy*torch.log(10e-8+estimed_policy)).sum(axis = 1) \
                + self.regularization*torch.square(utils.parameters_to_vector(parameters)).sum()
        rl_loss = rl_loss.sum()/true_value.shape[0]
        
        return rl_loss
    
class deepChessNN (Module):
    
    def __init__ (self, channels = 97, regularization = 0.01, lr = 0.01, betas = (0.9, 0.999), weight_decay = 0, tensorboard_dir = "./logs"):
        
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
    

        self.n_residuals = 19

        initial_convolution_block = [
            {
                "module": Conv2d,
                "in_channels": channels,
                "out_channels": 256,
                "kernel_size": (3,3),
                "padding": "same",
                "bias":False,
                "stride":(1,1)
            },
            {
                "module": BatchNorm2d,
                "num_features":256
            },
            {
                "module": ReLU
            }
        ]

        residual_block = [
            {
                "module": Conv2d,
                "in_channels": 256,
                "out_channels": 256,
                "kernel_size": (3,3),
                "padding": "same",
                "bias":False,
                "stride":(1,1)
            },
            {
                "module": BatchNorm2d,
                "num_features":256
            },
            {
                "module": ReLU
            },
            {
                "module": Conv2d,
                "in_channels": 256,
                "out_channels": 256,
                "kernel_size": (3,3),
                "padding": "same",
                "bias":False,
                "stride":(1,1)
            },
            {
                "module": BatchNorm2d,
                "num_features":256
            }
        ]

        self._initial_convolution = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in initial_convolution_block]
        self.initial_convolution = Sequential(*self._initial_convolution)
        
        self.residual_operations = ModuleList()
        for i in range(self.n_residuals):
            self._residual_operations = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in residual_block]
            self.residual_operations.append(Sequential(*self._residual_operations))
        
        value_projection_parameters = [
            {
                "module": Conv2d,
                "in_channels": 256,
                "out_channels": 2,
                "padding": "same",
                "kernel_size": (1,1),
                "bias":False,
                "stride":(1,1)
            },
            {
                "module": BatchNorm2d,
                "num_features":2
            },
            {
                "module": ReLU
            },
            {
                "module":Flatten
            },
            {
                "module": Linear,
                "in_features": 128,
                "out_features": 32,
                "bias":True
            },
            {
                "module": ReLU
            },
            {
                "module": Linear,
                "in_features": 32,
                "out_features": 1
            },
            {
                "module": Tanh
            }
        ]
        
        self._value_projection_operations = [x["module"](**
                                        dict(zip(list(x.keys())[1:], list(x.values())[1:]))
                                    ) for x in value_projection_parameters]
        self.value_projection_operations = Sequential(*self._value_projection_operations)
        
        moves_projection_parameters = [
            {
                "module": Conv2d,
                "in_channels": 256,
                "out_channels": 256,
                "kernel_size": (3,3),
                "padding": "same",
                "bias":False,
                "stride":(1,1)
            },
            {
                "module": BatchNorm2d,
                "num_features":256
            },
            {
                "module": ReLU
            },
            {
                "module": Conv2d,
                "in_channels": 256,
                "out_channels": 73,
                "kernel_size": (3,3),
                "padding": "same",
                "bias":False,
                "stride":(1,1)
            },
            {
                "module": Flatten
            },
            {
                "module": Softmax,
                "dim": 1
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
        self.log = False
        if tensorboard_dir is not None:
            self.writer = SummaryWriter(tensorboard_dir)
            self.nn_uid = random.randint(0, 10e5)
            current_date = datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%Y')
            self.nn_tb_tag = str("/".join([current_date, str(self.nn_uid)]))
            self.log = True
            self.i = 1
    
    def _register_loss(self, loss):
        
        """
            Function that register the loss value
            
            Input :
                loss : scalar value of the loss
            Output : None
        """
        
        # Register the loss
        self.losses.append(loss)
        
        # Only keeping 5 000 losses
        self.losses = self.losses[-5000:-1]+[self.losses[-1]]
        scalar_path = "/".join([str(self.nn_uid), "loss"])

        # Send the loss to tensorboard
        self.writer.add_scalar(
            scalar_path,
            loss.item(),
            self.i
        )
        self.i += 1

        self.writer.flush()
        
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

        ## Converting features in 8x8 format
        features = torch.concat([
                features,
                torch.zeros((features.shape[0], 64-features.shape[1]), device = features.device)
            ], axis = 1
        ).reshape((features.shape[0], 1, 8, 8))

        ## Concatening board and features
        input_tensor = torch.concat([board, features], axis = 1)

        ## 1.1. Encoding with CNN : Initial convolution

        output_tensor = self.initial_convolution(input_tensor)

        ## 1.2. Encoding with CNN : Residuals blocks
        for i in range(self.n_residuals):
            skip_connection = output_tensor.clone()
            output_tensor = self.residual_operations[i](output_tensor)
            output_tensor = ReLU()(output_tensor+skip_connection)
        
        # 2. Decoding
        
        ## 2.1 Value function
        value_function = output_tensor.clone()
        value_function = self.value_projection_operations(value_function)        
            
        ## 2.2 Probability distribution
        moves = output_tensor
        moves = self.moves_projection_operations(moves)            

        return value_function, moves
    
    def fit (self, x, y):
        
        """
            Forward pass
            
            Input :
                x : tuple containing the board matrice x[0] and the features x[0]
                y : tuple containing the current value function and the MTCS policy
            Output :
                tuple containing the value function and the next move probability distribution
        """

        self.train()

        # Zero grad
        self.optimizer.zero_grad()
        
        # Forward pass
        y_hat = self.forward(x)

        # Loss computation
        loss = self.loss(y_hat, y, self.parameters())
        if self.log == True:
            self._register_loss(loss)
        
        # Back propagation
        loss.backward()
        
        # Gradient descent
        self.optimizer.step()
        
    def predict(self, x):
        
        """
            Prediction
            
            Input : 
                x : tuple containing the board matrice x[0] and the features x[0]
        """
        
        # Prediction
        
        self.eval()

        with torch.no_grad():
            y_hat = self.forward(x)
            
        return y_hat
    
    def save(self, path):
        """
            Save a serialized version of the model
            
            Input : 
                path : path where to save the file
            Output : None
        """
        
        state = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'tb_dir': self.writer.get_logdir(),
            'nn_uid': self.nn_uid
        }
        
        torch.save(state, path)
    

def load(path, device = "cpu", tensorboard_dir = None):
    
    """
        Load a saved moed
        
        Input :
            path : path of the saved model
            tensorboard_dir : path of the log dir, if not specified, the original log dir is keeped
    """
    
    # Load the model
    state = torch.load(path, map_location = device)
    
    # Get log dir
    if tensorboard_dir is None:
        tensorboard_dir = state["tb_dir"]   

    if "nn_uid" in state.keys():
        nn_uid = state["nn_uid"]
    else:
        nn_uid = None

    # Instanciate the NN
    model = deepChessNN(tensorboard_dir=tensorboard_dir)
    if nn_uid is not None:
        model.nn_uid = nn_uid
    model.to(device)
    
    # Setting the parameters back
    model.load_state_dict(state['state_dict'])
    model.optimizer.load_state_dict(state['optimizer'])
    
    return model

torch_int = torch.int8
torch_float = torch.float32

def get_tensor(numpy_object, device = "cpu", dtype = torch_float):

    """
        Function to get a torch tensor from a numpy object
            input :
                numpy object : numpy object to convert to tensor
                device : device in which the tensor will be stored (cpu or cuda)
            ouput : 
                tensor object
    """

    tensor = torch.tensor(numpy_object, dtype = dtype)
    if device != 'cpu':
        tensor = tensor.to(device)

    return tensor

def get_lastest_model (model_folder, k):
    """
        get_lastest_model
        Get the lastest model of a current folder
        Input :
            model_folder : str, path of the folder containing the models
            k : number of models to get
        Ouput :
            path of the lastest models
    """
    
    file_list = glob.glob(f"{model_folder}/*.pt")
    file_name = [int(".".join(x.split("/")[-1].split(".")[0:-1])) for x in file_list]
    
    files = dict(zip(file_name, file_list))
    file_name.sort(reverse = True)
    
    output = []
    n_models = len(file_name)
    for i in range(k):
        if i < n_models:
            output.append(files[file_name[i]])
    
    return output