from __future__ import division

import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=1,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

# pretrained word vectors
parser.add_argument('-tie_weights', action='store_true',
                    help='Tie the weights of the encoder and decoder layer')
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-seed', default=9999, nargs='+', type=int,
                    help="Seed for deterministic runs.")

parser.add_argument('-log_interval', type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument('-save_every', type=int, default=-1,
                    help="Save every this interval.")
parser.add_argument('-uneven_batch', action='store_true',
                    help='Use imbalance mini_batches (default: all source sentences have the same length)')
opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

torch.manual_seed(opt.seed)

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    loss = 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    

    model.eval()
    for i in range(len(data)):
        # exclude original indices
        batch = data[i][:-1]
        outputs = model(batch)
        # exclude <s> from targets
        targets = batch[1][1:]
        loss, _ = memoryEfficientLoss(
                outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words


def trainModel(model, trainData, validData, dataset, optim):
    print(model)
    model.train()

    # Define criterion of each GPU.
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()

    def trainEpoch(epoch, batchOrder=None):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        
        if not batchOrder:
					batchOrder = torch.randperm(len(trainData))

        total_loss, total_words = 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        start = time.time()
        nSamples = len(trainData)
        for i in range(nSamples):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            batch = trainData[batchIdx][:-1]

            model.zero_grad()
            outputs = model(batch)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            loss, gradOutput = memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # Update the parameters.
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_tgt_words += num_words
            report_src_words += batch[0][1].data.sum()
            total_loss += loss
            total_words += num_words
            
            
        

            
            
            if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f;" +
                       "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                      (epoch, i+1, len(trainData),
                       
                       math.exp(report_loss / report_tgt_words),
                       report_src_words/(time.time()-start),
                       report_tgt_words/(time.time()-start),
                       time.time()-start_time))

                report_loss, report_tgt_words = 0, 0
                report_src_words = 0
                start = time.time()
            
            if opt.save_every > 0 and i % opt.save_every == -1 % opt.save_every :
							valid_loss = eval(model, criterion, validData)
							valid_ppl = math.exp(min(valid_loss, 100))
							print('Validation perplexity: %g' % valid_ppl)
							
							
							model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
							model_state_dict = {k: v for k, v in model_state_dict.items()
																	if 'generator' not in k}
							generator_state_dict = (model.generator.module.state_dict()
																			if len(opt.gpus) > 1
																			else model.generator.state_dict())
							checkpoint = {
									'model': model_state_dict,
									'generator': generator_state_dict,
									'dicts': dataset['dicts'],
									'opt': opt,
									'epoch': epoch,
									'iteration' : -1,
									'batchOrder' : None,
									'optim': optim
							}
							#  drop a checkpoint
							ep = float(epoch) - 1 + (i + 1) / nSamples
							checkpoint = {
									'model': model_state_dict,
									'generator': generator_state_dict,
									'dicts': dataset['dicts'],
									'opt': opt,
									'epoch': ep,
									'iteration' : i,
									'batchOrder' : batchOrder,
									'optim': optim
							}
							
							file_name = '%s_ppl_%.2f_e%.2f.pt'
							print('Writing to %s_ppl_%.2f_e%.2f.pt' % (opt.save_model, valid_ppl, ep))
							torch.save(checkpoint,
												 file_name
												 % (opt.save_model, valid_ppl, epoch))
        return total_loss / total_words

    for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
        print('')

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)

        #  (2) evaluate on the validation set
        valid_loss = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)

        #  (3) update the learning rate
        optim.updateLearningRate(valid_ppl, epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = (model.generator.module.state_dict()
                                if len(opt.gpus) > 1
                                else model.generator.state_dict())
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'iteration' : -1,
            'batchOrder' : None,
            'optim': optim
        }
        
			
        print('Writing to %s_ppl_%.2f_e%d.pt' % (opt.save_model, valid_ppl, epoch))
        torch.save(checkpoint,
                   '%s_ppl_%.2f_e%d.pt' 
                   % (opt.save_model, valid_ppl, epoch))


def main():
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    print("Done")
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"), balance=(not opt.uneven_batch))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"), balance=(not opt.uneven_batch))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.encoder_type == "text":
        encoder = onmt.Models.Encoder(opt, dicts['src'])
    elif opt.encoder_type == "img":
        encoder = onmt.modules.ImageEncoder(opt)
        assert("type" not in dataset or dataset["type"] == "img")
    else:
        print("Unsupported encoder type %s" % (opt.encoder_type))

    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    model = onmt.Models.NMTModel(encoder, decoder)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()
    
    model.generator = generator

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator
    
    # Weight tying :
    if opt.tie_weights:
			if len(opt.gpus) > 1:
				model.module.tie_weights()
			else:
				model.tie_weights()

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)
        
    

    optim.set_parameters(model.parameters())

    #~ if opt.train_from or opt.train_from_state_dict:
        #~ optim.optimizer.load_state_dict(
            #~ checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
