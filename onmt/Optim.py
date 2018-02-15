import torch.optim as optim
import torch
from copy import deepcopy
from itertools import chain
from torch.autograd import Variable
from collections import defaultdict, Iterable


#~ from torch.nn.utils import clip_grad_norm

def clip_grad_norm(parameters, max_norm, normalizer=1, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if normalizer > 1:
                p.grad.data.div_(normalizer)
            
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip_coef > 0:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm,
                 lr_decay=1, start_decay_at=None):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
    
    def normalize_grad(self, denom):
        
        parameters = list(filter(lambda p: p.grad is not None, self.params))
        

    def step(self, normalizer=1):
        "Compute gradients norm."
        if self.max_grad_norm:
            total_norm = clip_grad_norm(self.params, self.max_grad_norm, normalizer=normalizer)
        self.optimizer.step()
        return total_norm

    def updateLearningRate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl
        self.optimizer.param_groups[0]['lr'] = self.lr
        
    def get_learning_rate(self):
                
        return self.optimizer.param_groups[0]['lr']
                
    def set_learning_rate(self, lr):
                
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = lr
    
    def state_dict(self):
        
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        
        """Loads the optimizer state.
        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.optimizer.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            """Make a deep copy of value, casting all tensors to device of param."""
            if torch.is_tensor(value):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if any(tp in type(param.data).__name__ for tp in {'Half', 'Float', 'Double'}):
                    value = value.type_as(param.data)
                value = value.cuda(param.get_device()) if param.is_cuda else value.cpu()
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v
    
