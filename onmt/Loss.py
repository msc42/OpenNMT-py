import onmt
import onmt.modules
import torch.nn as nn
import torch, math
import torch.nn.functional as F

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
        

class NLLLoss(nn.Module):
    
    def __init__(self, num_classes, num_ignore=2, size_average=False):
        
        super(NLLLoss, self).__init__()
        self.size_average = size_average
        self.num_classes = num_classes
        self.padding_idx = onmt.Constants.PAD
        weight = torch.ones(num_classes)
        weight[onmt.Constants.PAD] = 0
        self.ignore_index = -100
        
        self.register_buffer('weight', weight)
        #~ self.crit = nn.NLLLoss(weight, size_average=False)
    
    def forward(self, input, target):
        loss = F.nll_loss(input, target, self.weight, self.size_average,
                          self.ignore_index)
        #~ loss = self.crit(input, target) 
        nll = loss.data[0]
        return loss, nll
        
class KLDivLoss(nn.Module):

    
    def __init__(self, num_classes, size_average=False, label_smoothing=0.1):
        super(KLDivLoss, self).__init__()
        self.size_average = size_average
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.padding_idx = onmt.Constants.PAD
        one_hot = torch.randn(1, num_classes)
        
        one_hot.fill_(label_smoothing / (num_classes - 2))
        one_hot[0][self.padding_idx] = 0
        one_hot[0][onmt.Constants.BOS] = 0
        self.register_buffer('one_hot', one_hot)
        
        self.confidence = 1 - label_smoothing
        

    def forward(self, input, target):
        """
        input: distribution output of a model (batch_size * num_classes)
        target: target index
        """  
        tdata = target.data
            
        # squeeze is a trick to know if mask has dimension or not
        mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze() 
        likelihood = torch.gather(input.data, 1, tdata.unsqueeze(1))
        likelihood.masked_fill_(tdata.eq(onmt.Constants.PAD), 0)
        tmp_ = self.one_hot.repeat(target.size(0), 1)
        tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
        if mask.dim() > 0:
            likelihood.index_fill_(0, mask, 0)
            tmp_.index_fill_(0, mask, 0)
       
        gtruth = torch.autograd.Variable(tmp_, requires_grad=False)
        
        loss = F.kl_div(input, gtruth, size_average=self.size_average)
        
        nll = - torch.sum(likelihood)
        
        #~ print(nll)
        
        return loss, nll


class LossFuncBase(nn.Module):

    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations
    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.
    Args:
        output_size: number of words in vocabulary()
    """
    
    def __init__(self, output_sizes):
        super(LossFuncBase, self).__init__()
        self.output_sizes = output_sizes
        self.padding_idx = onmt.Constants.PAD
    
    def _compute_loss(self, scores, targets):
        return NotImplementedError
    
    def forward(self, dists, targets, hiddens, **kwargs):
        """
        Compute the loss. Subclass must define this method.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError
        
        

class MemoryOptimizedNLLLoss(object):

    """
    Standard NMT Loss Computation.
    """
    def __init__(self, dicts, label_smoothing=0.0, shard_size=1, cuda=True):
        super(MemoryOptimizedNLLLoss, self).__init__()
        self.shard_split = shard_size
        self.label_smoothing = label_smoothing
        self.criterions = dict()
        self.dicts = dicts
        
        if self.label_smoothing > 0:
            print("Using KLDivLoss for label smoothing")
        
        for i in dicts:
            vocabSize = dicts[i].size()
            
            if label_smoothing > 0:
                crit = KLDivLoss(vocabSize, size_average=False, label_smoothing=label_smoothing)
            else:
                crit = NLLLoss(vocabSize, size_average=False)
            
            if cuda:
                crit = crit.cuda()
            
            if i == 0:
                print(crit)
                
            self.criterions[i] = crit
            
        
        
    
    def cuda(self):
        
        for i in self.criterions:
            self.criterions[i] = self.criterions[i].cuda()    
     
        
    def _compute_loss(self, scores, targets, setID):
        
        loss, nll = self.criterions[setID](scores, targets)
        
        return (loss, nll)
        
   
    def forward(self, outputs, targets, setID, generator=None, backward=False):
        """
        Compute the loss. Subclass must define this method.
        Args:
             
            outputs: the predictive output from the model. time x batch x vocab_size
                                                   or time x batch x hidden_size 
            target: the validate target to compare output with. time x batch
            generator: in case we want to save memory and 
            **kwargs(optional): additional info for computing loss.
        """
        batch_size = outputs.size(1)
        n_words = targets.data.ne(onmt.Constants.PAD).sum()
        
        outputs = torch.autograd.Variable(outputs.data, requires_grad=(backward))
                
        outputs_split = torch.split(outputs, self.shard_split)
        targets_split = torch.split(targets, self.shard_split)
        
        loss_data = 0
        for i, (outputs_t, target_t) in enumerate(zip(outputs_split, targets_split)):
            
            # compute the distribution 
            if generator is not None:
                dist_t = generator(outputs_t.view(-1, outputs_t.size(-1)))
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
            
        grad_outputs = None if outputs.grad is None else outputs.grad.data
        
        return loss_data, grad_outputs
