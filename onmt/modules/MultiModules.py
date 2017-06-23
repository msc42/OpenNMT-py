import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class MultiWordEmbedding(nn.Module):
	
	def __init__(self, opt, dicts):
		
		super(MultiWordEmbedding, self).__init__()
		self.inputSize = opt.word_vec_size
		self.moduleList = nn.ModuleList()
		
		for i in dicts:
			vocabSize = dicts[i].size()
			
			embedding = nn.Embedding(vocabSize,
															 self.inputSize,
															 padding_idx=onmt.Constants.PAD)
			self.moduleList.append(embedding)
		
		
		self.currentID = 0
		
	
	def switchID(self, idx):
		
		assert idx >= 0 and idx < len(self.moduleList)
		self.currentID = idx
		
	
	def forward(self, input):
		
		lookup = self.moduleList[self.currentID]
		
		embedding = lookup(input)
		
		return embedding


class MultiLinear(nn.Module):
	
	def __init__(self, inputSizes, outputSizes, bias=True):
		
		super(MultiLinear, self).__init__()
		
		self.inputSizes = inputSizes
		self.outputSizes = outputSizes
		
		assert len(self.inputSizes) == len(self.outputSizes)
		
		self.moduleList = nn.ModuleList()
		
		
		for i in xrange(len(self.inputSizes)):
			
			linear = nn.Linear(inputSizes[i], outputSizes[i], bias=bias)
			self.moduleList.append(linear)
			
		self.currentID = 0
		
	
	def switchID(self, idx):
	
		assert idx >= 0 and idx < len(self.moduleList)
		self.currentID = idx
		
	def forward(self, input):
	
		linear = self.moduleList[self.currentID]
		
		output = linear(input)
		
		return output

