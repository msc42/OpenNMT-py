import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable

from onmt.Loss import MemoryOptimizedNLLLoss

import onmt

# This module converts indices to one-hot vectors fast
class OneHot(nn.Module):
    
    def __init__(self, num_classes):
        super(OneHot, self).__init__()
        self.num_classes = num_classes
        one_hot = torch.randn(1, num_classes).zero_()
        
        # by pre-allocating the buffer, we can massively 
        # reduce the allocation time
        self.register_buffer('one_hot', one_hot)
        
    def forward(self, input):
        
        data = input.data
        original_size = data.size()
        # reshape 
        data = data.view(-1, 1)
        
        tmp_ = self.one_hot.repeat(data.size(0), 1)
        
        tmp_.scatter_(1, data, 1)
        
        tmp_ = tmp_.view(*(tuple(original_size) + (-1,)))
                
        return Variable(tmp_, requires_grad=False)

class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.
    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.
    The copy generator is an extended version of the standard
    generator that computse three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a
      word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """
    def __init__(self, opt, dicts):
        
        super(CopyGenerator, self).__init__()
        
        inputSize = opt.rnn_size
        self.inputSizes = [] 
        self.outputSizes = []
        
        self.one_hots = nn.ModuleList()
        
        for i in dicts:
            vocabSize = dicts[i].size()
            self.outputSizes.append(vocabSize)
            self.inputSizes.append(inputSize)
            
            self.one_hots.append(OneHot(vocabSize))
        
        
        
        # a function that decides the coefficients for the mixtures
        self.linear_copy = onmt.modules.MultiLinear(self.inputSizes, [1] * len(dicts))
        
        # compute regular softmax output
        self.linear = onmt.modules.MultiLinear(self.inputSizes, self.outputSizes)
    
    def switchID(self, tgtID):
        
        self.linear.switchID(tgtID)
        self.linear_copy.switchID(tgtID)
        
    def hardSwitchID(self, tgtID, reset_zero=False):
        self.linear_copy.hardSwitchID(tgtID, reset_zero=reset_zero)
        self.linear.hardSwitchID(tgtID, reset_zero=reset_zero)
    
    def forward(self, input, attn, src, return_log=True):
        
        """ First, we want to flatten the input """
        input = input.view(-1, input.size(-1))
        attn = attn.view(-1, attn.size(-1))
        batch_by_tlen_, slen = attn.size()
        batch_size = src.size(1)
        tlen = batch_by_tlen_ / batch_size
        
        
        # Compute the normal distribution by logits
        logits = self.linear(input)
        
        p_g = F.softmax(logits) # tlen * batch x vocab_size
        
        # Decide mixture coefficients
        copy = F.sigmoid(self.linear_copy(input))
        
        # Probibility of word coming from the generator distribution
        #~ p_g = torch.mul(prob,  1 - copy.expand_as(prob)) # tlen * batch x 1
        p_g = p_g.mul(1 - copy.expand_as(p_g))
        
        # Probibility of word coming from the copy pointer distribution
        p_c = torch.mul(attn, copy.expand_as(attn)) # tlen * batch x slen
        
        # create a mapping function (one hot vector) for the source positions
        #~ src_map = self.one_hots[self.linear.currentID](src) # slen x batch x vocab_size
        #~ batch_size = src_map.size(1)
        #~ print(p_g.size())
        #~ print(src.size())
        #~ print(p_c.size())
        
        
        # Idea: the ids of the source words are the same as the ids of the target words
        # So all we need to do is the scatter_add the corresponding probabilities to the output distribution
        # and avoid large matrices multiplication
        
        # In_place seems to work here, but if we modify p_g then error will appear
        p_g.scatter_add_(1, src.t().repeat(tlen, 1), p_c)
       
        
        # matrix multiplication:
        # b x tlen x slen  *  b x slen x vocabsize
        # transpose into tlen x b x slen
        #~ p_c = torch.bmm(mul_attn.view(-1 , batch_size, slen).transpose(0, 1),
                              #~ src_map.transpose(0, 1)).transpose(0, 1)
        #~ 
        #~ p_c = p_c.contiguous().view(-1, p_c.size(-1))
        
        #~ output = p_g + p_c
        
        # log probabilities
        
        output = p_g.clamp(min=1e-8)
        
        if return_log:
            output = torch.log(output)
                    
        return output
        
class MemoryOptimizedCopyLoss(MemoryOptimizedNLLLoss):
    
    def forward(self, batch, outputs, targets, setID, generator=None, backward=False):
        """
        Compute the loss. Subclass must define this method.
        Args:
             
            outputs: the predictive output from the model. time x batch x vocab_size
                                                   or time x batch x hidden_size 
            target: the validate target to compare output with. time x batch
            generator: in case we want to save memory and 
            **kwargs(optional): additional info for computing loss.
        """
        outputs, attns = outputs
        
        batch_size = outputs.size(1)
        n_words = targets.data.ne(onmt.Constants.PAD).sum()
        
        outputs_ = outputs
        attns_ = attns
        outputs = torch.autograd.Variable(outputs.data, requires_grad=(backward))
        attns = torch.autograd.Variable(attns.data, requires_grad=(backward))
                
        outputs_split = torch.split(outputs, self.shard_split)
        targets_split = torch.split(targets, self.shard_split)
        attns_split = torch.split(attns, self.shard_split)
        
        loss_data = 0
        for i, (outputs_t, attn_t, target_t) in enumerate(zip(outputs_split, attns_split, targets_split)):
            
            # compute the distribution 
            if generator is not None:
                src = batch[0][0]
                dist_t = generator(outputs_t, attn_t, src)
            else:
                dist_t = outputs_t.view(-1, outputs_t.size(-1))
           
            
            # actual loss function between the predictive distribution and target
            
            # flatten the distributions and the targets
            target_t = target_t.view(-1)
            loss_t, loss_data_t = self._compute_loss(dist_t, target_t, setID)

            loss_data += loss_data_t
            
            # backward from loss
            # note: we only compute the gradients w.r.t the outputs 
            if backward:
                loss_t.div(batch_size).backward()
                    
        if backward:
            variables = [outputs_, attns_]
            grads = [outputs.grad.data, attns.grad.data]
            torch.autograd.backward(variables, grads)
        
        return loss_data
