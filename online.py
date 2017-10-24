import os
from onmt.OnlineTranslator import TranslatorParameter,OnlineTranslator
import sys


filename="/model/model.conf"

t = OnlineTranslator(filename)

for line in sys.stdin:
    print t.translate(line)
