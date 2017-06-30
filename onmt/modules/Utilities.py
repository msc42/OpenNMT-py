import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottle(nn.Module):
	
		def __init__(self, m):
	
			super(Bottle, self).__init__()
			
			self.module = m() # we initialize the module using lambda 
			
		def forward(self, input):
			
			# input should be a 3D variable
			# B x L x H
			B = input.size(0)
			L = input.size(1)
			
			resized2D = input.view(B * L, -1)
			
			output = self.module(resized2D)
			
			resizedOut = output.contiguous().view(B, L, -1)
			
			return resizedOut


class LayerNorm(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
            
    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1).unsqueeze(1)
        sigma = torch.std(z, dim=1).unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out
