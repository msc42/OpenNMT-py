import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda

from onmt.modules import CopyGenerator

def update_opt(opt):
    """ For backward compatibility """ 
    
    if not hasattr(opt, 'copy_pointer'):
        opt.copy_pointer = False
    
    return opt

def build_model(opt, dicts, nSets):
    
    opt = update_opt(opt)
    
    encoder = onmt.Models.Encoder(opt, dicts['src'])
    decoder = onmt.Models.Decoder(opt, dicts['tgt'], nSets)
    
    if opt.copy_pointer == True:
        generator = CopyGenerator(opt, dicts['tgt'])
    else:
        generator = onmt.Models.Generator(opt, dicts['tgt'])
        
    print(generator)
        
    model = onmt.Models.NMTModel(encoder, decoder)
    
    if opt.share_embedding:
        model.shareEmbedding(dicts)
    if opt.share_projection:
        model.shareProjection(generator)
    
    return model, generator
