import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable
import onmt
#~ from onmt.Models import StackedLSTM


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
    
    def forward(self, inputs):
        
        """
        Input: the state of the translation model
            states (Variable): batch_size * hidden_size
                   optional: time * batch_size * hidden_size
        Output:
            predicted final reward given states (advantages)
            output (Variable): batch_size 
            optional: time * batch_size 
        """ 
        states = input['states']
        output = self.linear(states)
        
        # squeeze the final output
        output = output.squeeze(-1)
        
        return output


class RNNCritic(nn.Module):
    """
    A neural-encoder-decoder regression model as critic as in Ranzato et al 2015
    """
    def __init__(self, opt):
        
        super(RNNCritic, self).__init__()
        self.hidden_size = opt.rnn_size
        #~ self.lstm = onmt.Models.StackedLSTM(1, opt.rnn_size, opt.rnn_size, opt.dropout, opt.rnn_cell)
        self.lstm = nn.LSTM(opt.rnn_size, opt.rnn_size, 1)
        
        self.attn = onmt.modules.AttentionLayer(opt.rnn_size)
        
        self.linear = nn.Linear(opt.rnn_size, 1)
    
    def forward(self, inputs):
        """
        states (Variables): targetL x batch_size x dim
        context (Varibles): batch_size x sourceL x dim
        """
        
        """ detach the variables to avoid feedback loop """
        states = Variable(inputs['states'].data)
        context = Variable(inputs['context'].data)
        output, hidden = self.lstm(states) # targetL x batch_size x dim
        
        output_attn, attn = self.attn(output, context.transpose(0, 1))
        
        output = self.linear(output_attn).squeeze(-1)
        
        return output
    
