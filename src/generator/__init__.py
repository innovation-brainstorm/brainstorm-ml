

from generator.word_generator import WordGenerator
from generator.char_generator import CharGenerator
from generator.custom_generator import CustomGenerator



def get_generator(model_type,*args,**kwargs):
    if model_type=="WORD":
        return WordGenerator(*args,**kwargs)
    elif model_type=="CHAR":
        return CharGenerator(*args,**kwargs)
    elif model_type=="CUSTOM":
        return CustomGenerator(*args,**kwargs)
    else:
        raise Exception(f"no this model type: {model_type}")