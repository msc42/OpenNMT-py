import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import random



class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size
        dropout_value = opt.dropout 

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                           num_layers=opt.layers,
                           dropout=dropout_value,
                           bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            # Lengths data is wrapped inside a Variable.
            lengths = input[1].data.view(-1).tolist()
            emb = pack(self.word_lut(input[0]), lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):	
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size,
                               opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size
        self.input_size = input_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(0, 1))
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator, criterion):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.criterion = criterion
    
    # For sampling functions, we need to create an initial input
    # Which is vector of batch_size full of BOS    
    def make_init_input(self, src):
				if isinstance(src, tuple):
					src = src[0]
				batch_size = src.size(1)
				i_size = (1, batch_size)
				
				input_vector = src.data.new(*i_size).fill_(onmt.Constants.BOS)
				return Variable(input_vector, requires_grad=False)

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h
            
    def sample(self, input, max_length=100, argmax=True):
			
				
				src = input[0]
				# we don't care about tgt here
				
				enc_hidden, context = self.encoder(src)
				
				enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
				
				sampled = []
				
				# we start from the vector of <BOS>				
				input_t = self.make_init_input(src)
				
				# For input feeding initial output
				init_output = self.make_init_decoder_output(context)
				
				hidden = enc_hidden
				state = init_output
				
				tensor_check = None
				batch_size = context.size(1)
				
				lengths = torch.Tensor(batch_size).fill_(max_length).type_as(input_t.data)
				
				for t in xrange(max_length):
					# make a forward pass through the decoder
					state, hidden, attn_t = self.decoder(input_t, hidden, context, state)
					
					# from the state, compute the probability distribution
					# squeeze because state has size 1 x BS x H (due to concat)
					state = state.squeeze(0)
					output = self.generator(state) 
					
					if argmax:
						topv, topi = output.data.topk(1)
						
						input_t = Variable(topi.t())
					else: # Stochastic sampling
						input_t = output.exp().multinomial().t()

					# condition to stop the sampling procedure
					check = input_t.eq(onmt.Constants.EOS)
					if tensor_check is None:
						tensor_check  = check
					else:
						tensor_check += check
					
					
					lengths.masked_fill_(check.squeeze(0).data, t+1)
					
					# Every sampled input after EOS is just PAD
					input_t.masked_fill(tensor_check, onmt.Constants.PAD)

					sampled.append(input_t)
					
					# if all sentences are finished (reach EOS)
					#~ print(tensor_check.sum().data[0])
					if tensor_check.sum().data[0] == batch_size:
						break
						
				lengths = [lengths[i] for i in xrange(batch_size)]
					
				return sampled, lengths

    def forward(self, input, eval=False, mode='xe', teacher_forcing_ratio = 1):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        # Exclude <s> from targets for labels
        tgt_label = input[1][1:]
        
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
                      
        states = []
        
        outputs = []
        
        hidden = enc_hidden
        state = init_output
        batch_size = tgt.size(1) 
        length = tgt.size(0)
        
        
        # Here we loop over the input
        for t, input_t in enumerate(tgt.split(1)):
					
					use_teacher_forcing = random.random() <= teacher_forcing_ratio
					
					
					if not use_teacher_forcing and t > 1:
						topv, topi = outputs[t-1].data.topk(1)
						
						input_t = Variable(topi.t())
						
						
					
					state, hidden, attn_t = self.decoder(input_t, hidden, context, state)
					
					state = state.squeeze(0)
					states.append(state)
					
					# from the state, compute the probability distribution
					output = self.generator(state)
					
					outputs.append(output)
		
					# compute loss
					loss_t = self.criterion(output, tgt_label[t])
					
					if t == 0:
						loss = loss_t
					else:
						loss = loss + loss_t
						
        
        if not eval and ( mode =='xe' or mode == 'dad' ):
					loss.div(batch_size).backward()
				
        loss_data = loss.data[0]
				
        return loss_data, outputs, states
					
        
        


class Generator(nn.Module):
	
		
		def __init__(self, inputSize, dicts):
			
			super(Generator, self).__init__()
			
			self.inputSize = inputSize
			self.outputSize = dicts.size()
			
			self.net = nn.Sequential(
        nn.Linear(inputSize, self.outputSize),
        nn.LogSoftmax())

		def forward(self, input):
			return self.net(input)
