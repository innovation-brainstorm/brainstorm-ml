from utils.data_utils import split_data,read_csv
from dataset.text_dataset import TextDataset

class BaseGenerator(object):

    def __init__(self,data_or_filepath,output_dir):

        self.data_or_filepath=data_or_filepath
        self.output_dir=output_dir

        self.tokenizer=None
        self.model=None
        self.decoder=None


    def run(self):
        
        split_data(self.data_or_filepath,self.output_dir,0.8,0.2,0)

        train_dataset=TextDataset(root=self.output_dir,split="train")
        eval_dataset=TextDataset(root=self.output_dir,split="eval")

        self.train_tokenizer(train_dataset)

        self.train(train_dataset,eval_dataset)


    def train_tokenizer(self,train_dataset):
        pass

    def train(self):
        pass

    def generate(self,count):
        self.decoder.decode(self.model,self.tokenizer,count=count,max_length=100,top_p=0.6)

    