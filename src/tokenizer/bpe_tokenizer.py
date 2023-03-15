from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import decoders
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers import Encoding


class BpeTokenizer(object):

    def __init__(self):
        self.tokenizer=Tokenizer(BPE(unk_token="[UNK]",end_of_word_suffix="</w>"))
        self.tokenizer.pre_tokenizer=Whitespace()
        self.tokenizer.decoder=decoders.BPEDecoder()

        self.tokenizer.post_processor=TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS],2"),("[EOS]",3)]
        )

        self.tokenizer.enable_padding()

    def train(self,data):

            
        trainer=BpeTrainer(special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"],end_of_word_suffix="</w>")

        self.tokenizer.train(["../data/news_title/train.txt"],trainer)

        print(self.tokenizer.get_vocab_size())