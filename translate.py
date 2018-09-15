from __future__ import division

import onmt
import onmt.Markdown
import torch
import argparse
import math
import numpy
import os

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-src_lang',   default="en",
                    help='Source language')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-tgt_lang',   default="de",
                    help='Target language')                    
parser.add_argument('-ensemble_op',   default="sum",
                    help='Operator for ensemble decoding. Choices: sum/logsum')                    
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-print_nbest', action='store_true',
                    help='Output the n-best list instead of a single sentence')
parser.add_argument('-normalize', action='store_true',
                    help='To normalize the scores based on output length')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
parser.add_argument('-plot_attention', action="store_true",
                    help='Plot attention for decoded sequence')

def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None
    
def plotAttention(data, source, target, fname):
    
    import matplotlib.pyplot as plt # drawing heat map of attention weights
    import matplotlib.ticker as ticker
    #~ plt.rcParams['font.sans-serif']=['SimSun'] # set font family
    
    
    
    fig, ax = plt.subplots(figsize=(20, 8)) # set figure size
    cax = ax.matshow(data, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    #~ heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    #~ 
    X_label = [''] + [token.decode('utf-8') + "   " for token in source]
    Y_label = [''] + [token.decode('utf-8') + "   " for token in target]
    
    ax.set_xticklabels(X_label, rotation=90)
    ax.set_yticklabels(Y_label)
    
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax.grid(True)
    
    file_name = fname + ".png"
    #~ 
    fig.savefig(file_name) 
    print ("[INFO] Saving attention heatmap to %s" % file_name)
    plt.close(fig)    # close the figure
    
# attn should have size n_target * n_source
def printAttention(attns, srcBatch, predBatch, basename, baseID, directory="temp/"):
    
    if not os.path.exists(directory + basename):
        os.makedirs(directory + basename)
    
    sentID = baseID
    for attnBeams, srcSent, predBeams in zip(attns, srcBatch, predBatch):
        
        sentID += 1 
        # only get attn and pred from top of the beam
        attn = attnBeams[0]
        pred = predBeams[0]# list of target words 
        n_target = attn.size(0)
        src = srcSent # list of source words
        
        # because EOS is trimmed when convert from id to word
        # so we might want to restore it for displaying
        if len(pred) == n_target - 1:
            pred = pred + [onmt.Constants.EOS_WORD]
        
        # sanity check
        c1 = (len(pred) == attn.size(0))
        c2 = (len(src) == attn.size(1))
        
        if not (c1 and c2):
            continue
                
        # checking if the sum of attention must be 1.00
        sum_row = float("{0:.2f}".format(torch.sum(attn[0])))
        assert sum_row== 1.00, "Attention weights for one target token w.r.t context must sum to 1.00, currently is %.2f" % sum_row
        
        # convert this crap to numpy
        data = attn.cpu().numpy()
        
        fname = directory + basename + "/" + str(sentID)
        
        # plot the attention
        plotAttention(data, src, pred, fname)


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    
    # Always pick n_best
    opt.n_best = opt.beam_size
    
    if opt.plot_attention:
        try:
            import matplotlib   
            matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
        except ImportError:
            print("[ERROR] Matplotlib is required to plot attention maps")
            return
        
    translator = onmt.Translator(opt)
    
    # for graph plotting
    basename = opt.output + ".attn"

    outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None

    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()

    for line in addone(open(opt.src)):
        if line is not None:
            srcTokens = line.split()
            srcBatch += [srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, predScore, goldScore, goldWords, attn = translator.translate(srcBatch,
                                                               tgtBatch)
        
        if opt.normalize:
            predBatch_ = []
            predScore_ = []
            for bb, ss in zip(predBatch, predScore):
                    ss_ = [s_/numpy.maximum(1., len(b_)) for b_,s_ in zip(bb,ss)]
                    sidx = numpy.argsort(ss_)[::-1]
                    predBatch_.append([bb[s] for s in sidx])
                    predScore_.append([ss_[s] for s in sidx])
            predBatch = predBatch_
            predScore = predScore_
                                                              
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if tgtF is not None:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += goldWords
            
        # plot the attention heat map if needed
        if opt.plot_attention:
            printAttention(attn, srcBatch, predBatch, basename, count)
            

        for b in range(len(predBatch)):
            # Pred Batch always have n-best outputs  
            count += 1
            # Best sentence = having highest log prob

            if not opt.print_nbest:
                outF.write(" ".join(predBatch[b][0]) + '\n')
                outF.flush()
            else:
                for n in range(opt.n_best):
                    idx = n
                    #~ if opt.verbose:
                    print("%d ||| %s ||| %.6f" % (count-1, " ".join(predBatch[b][idx]), predScore[b][idx]))
                    outF.write("%d ||| %s ||| %.6f\n" % (count-1, " ".join(predBatch[b][idx]), predScore[b][idx]))
                    outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % goldScore[b])
                print('')

        srcBatch, tgtBatch = [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(translator.beam_accum, open(opt.dump_beam, 'w'))


if __name__ == "__main__":
    main()
