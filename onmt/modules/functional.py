import torch
from torch.autograd import Variable


def to_one_hot(y, depth=None):
    r"""
    Takes integer with n dims and converts it to 1-hot representation with n + 1 dims.
    The n+1'st dimension will have zeros everywhere but at y'th index, where it will be equal to 1.
    Args:
        y: input integer (IntTensor, LongTensor or Variable) of any shape
        depth (int):  the size of the one hot dimension
    Examples::
        >>> to_one_hot(torch.arange(0, 5) % 3)
         1  0  0
         0  1  0
         0  0  1
         1  0  0
         0  1  0
        [torch.FloatTensor of size 5x3]
        >>> to_one_hot(torch.arange(0, 5) % 3, depth=5)
         1  0  0  0  0
         0  1  0  0  0
         0  0  1  0  0
         1  0  0  0  0
         0  1  0  0  0
        [torch.FloatTensor of size 5x5]
        >>> to_one_hot(torch.arange(0, 6).view(3,2) % 3)
        (0 ,.,.) =
          1  0  0
          0  1  0
        (1 ,.,.) =
          0  0  1
          1  0  0
        (2 ,.,.) =
          0  1  0
          0  0  1
        [torch.FloatTensor of size 3x2x3]
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_tensor)) + 1
    
    size_ = (y_tensor.size(0), depth)
    
    """ don't forget the asterisk """
    y_one_hot = y_tensor.new(*size_).zero_()
    
    y_one_hot.scatter_(1, y_tensor, 1)
        
    y_one_hot = y_one_hot.view(*(tuple(y.size()) + (-1,)))
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
    
    
