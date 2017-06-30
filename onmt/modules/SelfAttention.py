import torch
import torch.nn as nn
import torch.nn.init as init
import onmt.modules
import numpy as np
from torch.autograd import Variable
import math

	

# This module takes as input:
# - A set of queries (  B x nQueries x H )
# - Set of values ( B x L x H )
# - Set of keys ( B x L x H )
# Produce the output:
# - layernorm(queries + attn(Q, V, K)
# Using simple dot attention 
class MultiHeadedAttention(nn.Module):
	
	def __init__(self, dim, heads=1, p=0, norm=True):
		
		super(MultiHeadedAttention, self).__init__()
		self.p = p
		self.dim = dim
		
		assert dim % heads == 0 # To avoid headaches
		self.d_k = dim // heads 
		self.heads = heads
		
		lin = lambda : nn.Linear(dim, dim)
		# projectors 
		self.project_keys = onmt.modules.Bottle(lin)
		self.project_values = onmt.modules.Bottle(lin)
		self.project_queries = onmt.modules.Bottle(lin)
		
		sm = lambda : nn.Softmax()
		self.sm = onmt.modules.Bottle(sm)
		
		# dropout layers
		self.dropout = nn.Dropout(p)
		self.res_dropout = nn.Dropout(p)
		
		# layer norm layer
		
		if norm:
			ln = lambda : onmt.modules.LayerNorm(dim)
			self.layer_norm = onmt.modules.Bottle(ln)
		self.norm = norm
		
	def forward(self, key, value, query, mask=None):
				
		def shape(x):
			return x.view(x.size(0), x.size(1), self.heads, self.d_k).transpose(1, 2) \
							.contiguous().view(x.size(0) * self.heads, x.size(1), self.d_k)
					
									
		def unshape(x):
			return x.view(x.size(0)//self.heads, self.heads, x.size(1), x.size(2))
			
		def smash(x):
			return x.view(x.size(0) * self.heads, x.size(2), x.size(3))
			
		key_up = shape(self.project_keys(key)) # B * H x Lt x D_k
		value_up = shape(self.project_values(value)) # B * H x Lt x D_k
		query_up = shape(self.project_queries(query)) # B * H x Ls x D_k
		
		# compute attention values for all of them at once
		scaled = torch.bmm(query_up, key_up.transpose(1, 2)) 
		scaled = scaled / math.sqrt(self.d_k)
		
		# scaled should have size ( bsize * heads, Ls, Lt )
		
		if mask is not None: # Mask should be B x Lt 
			scaled = unshape(scaled) # B x H x Ls x Lt
			mask = mask.unsqueeze(1).expand_as(scaled) # Mask size is 
			scaled = scaled.masked_fill(Variable(mask), -float('inf'))
			scaled = smash(scaled) 
		
		# the bottle softmax computes sm on the last dimension
		# which is Lt, which is the dim we need	
		attn = self.dropout(self.sm(scaled)) # B * H x Ls x Lt
		
		#~ print(attn)
		
		out = torch.bmm(attn, value_up) # B * H x Ls x D_k
		
		out = out.view(query.size(0), self.heads, query.size(1), self.d_k) \
						 .transpose(1, 2).contiguous() \
						 .view(query.size(0), query.size(1), self.heads * self.d_k)
						 
		# residual then layer norm
		#~ res = self.res_dropout(out) + query
		res = out + query
		
		if self.norm:
			res = self.layer_norm(res)
		
		return res, attn
		
		
		
