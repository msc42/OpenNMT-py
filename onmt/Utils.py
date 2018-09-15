import subprocess
import torch

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def compute_score(score, samples, ref, tgtDict, batch_size, average=True):
        
    # probably faster than gpu ?
    sdata = samples.data.cpu().t().tolist()
    rdata = ref.data.cpu().t().tolist()
        
    s = torch.Tensor(batch_size)
    
    for i in range(batch_size):
        
        sampledIDs = sdata[i]
        refIDs = rdata[i]
        
        sampledWords = tgtDict.convertToLabels(sampledIDs, onmt.Constants.EOS)
        refWords = tgtDict.convertToLabels(refIDs, onmt.Constants.EOS)
        
        # note: the score function returns a tuple 
        s[i] = score(refWords, sampledWords)[0]
        
    s = s.cuda()
        
    return s


# Update the options for backward compatibility
def update_opt(model_opt):
    
    if not hasattr(model_opt, 'rnn_cell'):
        model_opt.rnn_cell = 'lstm'
    
    if not hasattr(model_opt, 'share_projection'):
        model_opt.share_projection = False
    
    return model_opt
