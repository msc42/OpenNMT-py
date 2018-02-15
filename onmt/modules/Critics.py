import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable


class MLPCritic(nn.Module):
    
    """
    A neural-net regression model as critic as in Ranzato et al 2015
    """
    
    def __init__(self, opt):
        
        super(MLPCritic, self).__init__()
        self.hidden_size = opt.rnn_size
        self.linear = nn.Sequential(
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(),
                            nn.Linear(self.hidden_size, 1))
    
    def forward(self, states):
        
        """
        Input: the state of the translation model
            states (Variable): batch_size * hidden_size
                   optional: time * batch_size * hidden_size
        Output:
            predicted final reward given states (advantages)
            output (Variable): batch_size 
            optional: time * batch_size 
        """ 
        output = self.linear(states)
        
        # squeeze the final output
        output = output.squeeze(-1)
        
        return output


class NEDCritic(nn.Module):
    """
    A neural-encoder-decoder regression model as critic as in Ranzato et al 2015
    """
    def __init__(self, hidden_size):
        
        super(NEDCritic, self).__init__()
        #~ self.hidden_size = hidden_size
        #~ self.linear = nn.Sequential(
                            #~ nn.Linear(self.hidden_size, self.hidden_size),
                            #~ nn.ReLU(),
                            #~ nn.Linear(self.hidden_size, 1))
