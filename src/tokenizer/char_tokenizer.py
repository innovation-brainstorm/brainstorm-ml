import json
import pickle
import os

class Encoding(object):
    def __init__(self,ids):
        self.ids=ids

# TODO: inherit from tokenizer
class CharacterTokenizer(object):

    def __init__(self,token_sep="[SEP]",token_unk="[UNK]",token_pad="[PAD]",token_bos="[BOS]",token_eos="[EOS]"):

        self.id_to_token=[]
        self.token_id_dict={}

        self.token_unk=token_unk
        self.token_pad=token_pad
        self.token_bos=token_bos
        self.token_eos=token_eos
        self.token_sep=token_sep

        self.special_tokens=[token_pad,token_sep,token_unk,token_bos,token_eos]

        self._init_special_tokens(self.special_tokens)

        self.vocab_size=0


    def get_vocab_size(self):
        return self.vocab_size

    def _init_special_tokens(self,special_tokens):
        for token in special_tokens:
            _id=len(self.id_to_token)
            self.id_to_token.append(token)
            self.token_id_dict[token]=_id

    def encode(self,text):
        encoding=[]
        encoding.append(self.token_id_dict[self.token_bos])

        for char in text:
            encoding.append(self.token_id_dict.get(char,self.token_id_dict[self.token_unk]))

        encoding.append(self.token_id_dict[self.token_eos])

        return Encoding(encoding)

    def encode_batch(self,samples):
        encodings=[]
        for sample in samples:
            encodings.append(self.encode(sample))

        return self._pad(encodings)


    def _pad(self,encodings):
        max_length=max([len(encoding.ids) for encoding in encodings])

        for encoding in encodings:
            padding_count=max_length-len(encoding.ids)

            if padding_count>0:
                encoding.ids+=[self.token_id_dict[self.token_pad]]*padding_count

        return encodings

    
    def decode(self,token_ids):
        decoding=[]
        for token_id in token_ids:

            token=self.id_to_token[token_id]
            if token not in self.special_tokens:
                decoding.append(token)

        return "".join(decoding)


    def decode_batch(self,sequences):
        results=[]
        for token_ids in sequences:
            results.append(self.decode(token_ids))

        return results

    
    def train(self,data):
        for row in data:
            for char in row:
                if char not in self.token_id_dict:
                    _id=len(self.id_to_token)
                    self.id_to_token.append(char)
                    self.token_id_dict[char]=_id

        self.vocab_size=len(self.id_to_token)

    def token_to_id(self,token):
        return self.token_id_dict[token]
    
    def load(self,dir):
        filename=self._get_file_name(dir)
        with open(filename, 'r') as f:
            tmp_dict = json.load(f)

        self.__dict__.update(tmp_dict)
        return self


    def save(self,dir):
        filename=self._get_file_name(dir)
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)

    def _get_file_name(self,dir):
        return os.path.join(dir,"tokenizer.json")

    




