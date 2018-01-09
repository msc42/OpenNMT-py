from __future__ import division

import onmt
import onmt.Markdown
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from onmt.trainer.Evaluator import Evaluator
from onmt.trainer.XETrainer import XETrainer
from onmt.trainer.SelfCriticalTrainer import SCSTTrainer
import math
import time

from onmt.trainer.Evaluator import Evaluator

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
parser.add_argument('-adapt_src', default='',
                    help="""source language to adapt""")
parser.add_argument('-adapt_tgt', default='',
                    help="""target language to adapt""")
parser.add_argument('-override', action='store_true',
                    help="""Overwrite the save file to reduce space consumption""")
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
parser.add_argument('-epochs', type=int, default=15,
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
parser.add_argument('-reinforce', action='store_true',
                    help="""Using reinforcement learning""")
parser.add_argument('-reinforce_metrics', default='gleu',
                    help="Type of metrics to use. Options are [gleu|hit].")    
parser.add_argument('-hit_alpha', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')

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
parser.add_argument('-reset_optim', action='store_true',
                    help="""reset the optimization""")
# pretrained word vectors

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

# For multilingual configs
parser.add_argument('-share_rnn_enc', action='store_true',
                    help="""Share Rnn Encoder""")
parser.add_argument('-share_rnn_dec', action='store_true',
                    help="""Share Rnn Decoder""")
parser.add_argument('-share_embedding', action='store_true',
                    help="""Share embedding between same language in enc and dec""")
parser.add_argument('-share_attention', action='store_true',
                    help="""Share attentional modules between pair""")
opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

torch.manual_seed(opt.seed)



def NMTCriterion(dicts):
    
    crits = dict()
    for i in dicts:
        vocabSize = dicts[i].size()
        
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        if opt.gpus:
            crit.cuda()
        
        crits[i] = crit
    
    return crits


def trainModel(model, trainSets, validSets, dataset, optim, evaluator):
    print(model)
    model.train()

    # Define criterion of each target language.
    criterions = NMTCriterion(dataset['dicts']['tgt'])
    setIDs = dataset['dicts']['setIDs']

    start_time = time.time()

    def trainEpoch(epoch, batchOrder=None):

        # Shuffle mini batch order.
        if not batchOrder:
            batchOrder = dict()
            for i in trainSets:
                batchOrder[i] = torch.randperm(len(trainSets[i]))

        total_loss, total_words = dict(), dict()
        report_loss, report_tgt_words = dict(), []
        report_src_words = []
        start = time.time()
        
        for i in trainSets:
            total_loss[i] = 0
            total_words[i] = 0
            report_loss[i] = 0
            report_tgt_words.append(0)
            report_src_words.append(0)
        
        dataSizes = [len(trainSets[i]) for i in trainSets]
        nSamples = sum(dataSizes)
        
        # In order to make sets sample randomly,
        # We create a distribution over the data size
        # In the future we can manipulate this distribution 
        # to create biased sampling when training
        sampleDist = torch.Tensor(len(setIDs))
        iterators = dict()
        for i in xrange(len(setIDs)):
                    sampleDist[i] = len(trainSets[i])
                    iterators[i] = -1
        sampleDist = sampleDist / torch.sum(sampleDist)

        for i in range(nSamples):
                        
            sampledSet = -1
            while True:
                # if the sampled set is full then we re-sample 
                # to ensure that in one epoch we read each example once
                sampledSet = int(torch.multinomial(sampleDist, 1)[0])
                if iterators[sampledSet] + 1 < dataSizes[sampledSet]:
                    break
            
            iterators[sampledSet] += 1 
            
            # Get the batch index from batch order
            batchIdx = batchOrder[sampledSet][iterators[sampledSet]] if epoch > opt.curriculum else iterators[sampledSet]
            
            # Get the batch
            batch = trainSets[sampledSet][batchIdx][:-1]
            batch_size = batch[1].size(1)
            
            # And switch the model to the desired language mode
            model.switchLangID(setIDs[sampledSet][0], setIDs[sampledSet][1])
            model.switchPairID(sampledSet)
            
            # Do forward to the newly created graph
            model.zero_grad()
            outputs, hiddens = model(batch)
            
            # Exclude <s> from targets.
            targets = batch[1][1:]
            # The criterion is for the target language side
            criterion = criterions[setIDs[sampledSet][1]]
            
            loss_v = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss = loss_v.data[0]
            
            loss_v.div(batch_size).backward()
            
                        
            # Update the parameters.
            optim.step()

            # Statistics for the current set
            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss[sampledSet] += loss
            report_tgt_words[sampledSet] += num_words
            report_src_words[sampledSet] += batch[0][1].data.sum()
            total_loss[sampledSet] += loss
            total_words[sampledSet] += num_words

            # Logging information
            if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                avgTrainLoss = averagePPL(report_loss, report_tgt_words)
                logOut = ("Epoch %2d, %5d/%5d; ; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed; ppl: %6.2f; lr: %.6f" %
                                (epoch, i+1, nSamples,
                                 sum(report_src_words)/(time.time()-start),
                                 sum(report_tgt_words)/(time.time()-start),
                                 time.time()-start_time,
                                 avgTrainLoss,
                                 optim.get_learning_rate()))
                                 
                for j in xrange(len(setIDs)):
                    
                    report_loss[j] = 0
                    report_tgt_words[j] = 0
                    report_src_words[j] = 0
                    
                print(logOut)
                start = time.time()    
                            
                
            # Saving checkpoints with validation perplexity
            if opt.save_every > 0 and i % opt.save_every == -1 % opt.save_every :
                valid_losses = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
                valid_ppl = [math.exp(min(valid_loss, 100)) for valid_loss in valid_losses]
                #~ valid_ppl = " ".join([str(math.exp(min(valid_loss, 100))) for valid_loss in valid_losses])
                for i in xrange(len(setIDs)):
                    setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][i])
                    print('Validation perplexity for set %s : %g' % (setLangs, valid_ppl[i]))
                
                
                avgDevPpl = sum(valid_ppl) / len(valid_ppl)
                model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                else model.state_dict())
                model_state_dict = {k: v for k, v in model_state_dict.items()
                                                        if 'generator' not in k}
                generator_state_dict = (model.generator.module.state_dict()
                                                                if len(opt.gpus) > 1
                                                                else model.generator.state_dict())
                #  drop a checkpoint

                ep = float(epoch) - 1.0 + float(i + 1.0) / float(nSamples)

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
                #~ valid_ppl = "_".join([("%.2f" % math.exp(min(valid_loss, 100))) for valid_loss in valid_losses])
                print('Writing to %s_ppl_%.2f_e%.2f.pt' % (opt.save_model, avgDevPpl, ep))
                torch.save(checkpoint,
                                     file_name
                                     % (opt.save_model, avgDevPpl, ep))
        return [total_loss[j] / total_words[j] for j in xrange(len(setIDs))]
        
    #~ valid_losses = eval(model, criterions, validSets, setIDs)
    valid_losses = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
    valid_ppl = [math.exp(min(valid_loss, 100)) for valid_loss in valid_losses]
    for i in xrange(len(setIDs)):
        setLangs = "-".join(lang for lang in dataset['dicts']['setLangs'][i])
        print('Validation perplexity for set %s : %g' % (setLangs, valid_ppl[i]))
                
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
        print('')

        #  (1) train for one epoch on the training set
        train_losses = trainEpoch(epoch)
        train_ppl = [math.exp(min(train_loss, 100)) for train_loss in train_losses]
        for i in xrange(len(setIDs)):
                    print('Training perplexity for set %d : %g' % (i, train_ppl[i]))

        #  (2) evaluate on the validation set
        valid_losses = evaluator.eval_perplexity(validSets, criterions, setIDs=setIDs)
        valid_ppl = [math.exp(min(valid_loss, 100)) for valid_loss in valid_losses]
        avgDevPpl = sum(valid_ppl) / len(valid_ppl)
        for i in xrange(len(setIDs)):
            print('Validation perplexity for set %d : %g' % (i, valid_ppl[i])) 
        #  (3) update the learning rate
        #~ optim.updateLearningRate(valid_ppl, epoch)

        model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                            else model.state_dict())
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
                            
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
        
                
        #~ valid_ppl = "_".join([("%.2f" % math.exp(min(valid_loss, 100))) for valid_loss in valid_losses])
        file_name = '%s_ppl_%.2f_e%d.pt'
        print('Writing to %s_ppl_%.2f_e%d.pt' % (opt.save_model, avgDevPpl, epoch))
        torch.save(checkpoint,
                                     file_name
                                     % (opt.save_model, avgDevPpl, epoch))


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
    
    dicts = dataset['dicts']
    nSets = dicts['nSets']
    print(' * Vocabulary sizes: ')
    for lang in dicts['langs']:
            print(' * ' + lang + ' = %d' % dicts['vocabs'][lang].size())

    trainSets = dict()
    validSets = dict()
    for i in xrange(nSets):
      trainSets[i] = onmt.Dataset(dataset['train']['src'][i],
                             dataset['train']['tgt'][i], opt.batch_size, opt.gpus)
            
      validSets[i] = onmt.Dataset(dataset['valid']['src'][i],
                             dataset['valid']['tgt'][i], opt.batch_size, opt.gpus)
      
      print(' * number of training sentences for set %d: %d' %
          (i, len(dataset['train']['src'][i])))
        

    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')
    
    
    encoder = onmt.Models.Encoder(opt, dicts['src'])
    decoder = onmt.Models.Decoder(opt, dicts['tgt'], nSets)
    generator = onmt.Models.Generator(opt, dicts['tgt'])

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
        opt.start_epoch = int(math.floor(checkpoint['epoch'] + 1))

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator
    
    if opt.share_embedding:
            model.shareEmbedding(dicts)

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    elif not opt.reset_optim and 'optim' in checkpoint:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
    else:
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )

    optim.set_parameters(model.parameters())
    optim.set_learning_rate(opt.learning_rate)
    
    

    #~ if opt.train_from or opt.train_from_state_dict:
        #~ optim.optimizer.load_state_dict(
            #~ checkpoint['optim'].optimizer.state_dict())
    
    if opt.train_from or opt.train_from_state_dict:
        del checkpoint # to save memory

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    
    if len(opt.adapt_src) > 0 and len(opt.adapt_tgt) > 0:
    
        # find the source and target ID of the pair we need to adapt
        srcID = dataset['dicts']['srcLangs'].index(opt.adapt_src)
        tgtID = dataset['dicts']['tgtLangs'].index(opt.adapt_tgt)
    
        setIDs = dataset['dicts']['setIDs']
        
        # find the pair ID that we need to adapt
        pairID = -1
        for i, sid in enumerate(setIDs):
            if sid[0] == srcID and sid[1] == tgtID:
                pairID = i
                break
                
        if pairID == -1:
            pairID = None
    
    else:
        srcID = None
        tgtID = None
        pairID = None
    
    # convert string to IDs for easier manipulation
    opt.adapt_src = srcID
    opt.adapt_tgt = tgtID 
    opt.pairID = pairID
    
    evaluator = Evaluator(model, dataset, opt, cuda=(len(opt.gpus) >= 1))
    
    if opt.reinforce:
        trainer = SCSTTrainer(model, trainSets, validSets, dataset, optim, evaluator, opt)
    else:
        trainer = XETrainer(model, trainSets, validSets, dataset, optim, evaluator, opt)
    
    trainer.run()

    #~ trainModel(model, trainSets, validSets, dataset, optim, evaluator)


if __name__ == "__main__":
    main()
