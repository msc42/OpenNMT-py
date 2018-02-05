import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda

from onmt.modules import CopyGenerator

def build_model(opt, dicts, nSets):
    
    encoder = onmt.Models.Encoder(opt, dicts['src'])
    decoder = onmt.Models.Decoder(opt, dicts['tgt'], nSets)
    
    if opt.copy_pointer:
        generator = CopyGenerator(opt, dicts['tgt'])
    else:
        generator = onmt.Models.Generator(opt, dicts['tgt'])
        
    model = onmt.Models.NMTModel(encoder, decoder)
    
    return model, generator
