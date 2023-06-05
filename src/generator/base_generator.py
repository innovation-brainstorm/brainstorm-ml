import pathlib
import os
from utils.data_utils import split_data,write_txt


class BaseGenerator(object):

    def __init__(self,data_or_filepath,data_dir):

        self.data_or_filepath=data_or_filepath
        self.data_dir=data_dir

        self.tokenizer=None
        self.model=None
        self.decoder=None


    def run(self):
        
        split_data(self.data_or_filepath,self.data_dir,0.8,0.2,0)

        train_dataset,eval_dataset=self.build_dataset()

        self.train_tokenizer(train_dataset)

        self.train(train_dataset,eval_dataset)

    
    def build_dataset(self,dir):
        pass


    def train_tokenizer(self,train_dataset):
        pass

    def train(self):
        pass

    def generate(self):
        pass

    def save(self,dir):
        p = pathlib.Path(dir)
        p.mkdir(parents=True, exist_ok=True)
        write_txt(os.path.join(dir,"model_type"),[self.model_type])
        self._save(dir)





    