import onmt
import onmt.modules


class TranslatorParameter(object):

    def __init__(self):

        self.model = "";
        self.src = "<stdin>";
        self.src_img_dir = "";
        self.tgt = "";
        self.output = "<stdout>";
        self.beam_size = 5
        self.batch_size = 1
        self.max_sent_length = 100
        self.dump_beam = ""
        self.n_best = self.beam_size
        self.replace_unk = False
        self.gpu = -1;
        self.cuda = 0;
        

class OnlineTranslator(object):
    def __init__(self,model):
        opt = TranslatorParameter()
        opt.model = model
        self.translator = onmt.Translator(opt)
    

    def translate(self,input):
              predBatch, predScore, goldScore = self.translator.translate([input],[])
              return " ".join(predBatch[0][0])
  

