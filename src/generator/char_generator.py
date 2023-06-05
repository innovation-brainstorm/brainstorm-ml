import os
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from generator.base_generator import BaseGenerator
from utils.data_utils import transform
from tokenizer.char_tokenizer import CharacterTokenizer

from model.lstm_lm import LSTMLanguageModel
from decoding.top_p_decoding import TopPDecoding
from dataset.text_dataset import TextDataset
from utils.early_stop import EarlyStop

import logging
logger=logging.getLogger(__name__)

class CharGenerator(BaseGenerator):

    model_type="CHAR"

    hidden_size=125
    n_layers=1
    epochs=50
    batch_size=16
    learning_rate=0.0001
    

    def __init__(self,data_or_filepath,data_dir,model_dir=None):

        self.data_or_filepath=data_or_filepath
        self.data_dir=data_dir
        if model_dir:
            self.model=torch.load(os.path.join(model_dir,"pytorch_model.bin"))
            self.tokenizer=CharacterTokenizer().load(model_dir)
        else:
            self.tokenizer=CharacterTokenizer()
            self.model=None

        self.decoder=TopPDecoding()

    def build_dataset(self):
            
        train_dataset=TextDataset(root=self.data_dir,split="train",max_count=5000)
        eval_dataset=TextDataset(root=self.data_dir,split="eval",max_count=1000)

        return train_dataset,eval_dataset

    def train_tokenizer(self,train_data:Dataset):
        self.tokenizer.train(train_data.data)

        v=self.tokenizer.get_vocab_size()
        logger.info(f"vocab count: {v}")


    def train(self,train_data:Dataset,eval_data:Dataset):


        v=self.tokenizer.get_vocab_size()


        train_dataloader=DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
        eval_dataloader=DataLoader(eval_data,batch_size=self.batch_size,shuffle=True)

        model=LSTMLanguageModel(v,self.hidden_size,v,self.n_layers)
        model.train()

        loss_fn=nn.CrossEntropyLoss(ignore_index=0)
        optimizer=torch.optim.Adam(model.parameters(),lr=self.learning_rate)

        early_stop=EarlyStop(patience=2,delta=0)

        train_loss_list=[]
        val_loss_list=[]
        for i in range(self.epochs):
            logger.info(f"Epoch:{i}/{self.epochs}..........")
            train_loss=self.train_loop(train_dataloader,model,loss_fn,optimizer)
            val_loss=self.test_loop(eval_dataloader,model,loss_fn)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            early_stop(val_loss)
            if early_stop.early_stop:
                logger.info(f"Early Stop training")
                break

            #TODO: save model

        self.model=model



        
    def train_loop(self,dataloader,model,loss_fn,optimizer):

        print_every=10

        size=len(dataloader.dataset)
        print_loss=0
        running_loss=0

        for batch,texts in enumerate(dataloader):

            encodings=self.tokenizer.encode_batch(texts)
            encodings=[e.ids for e in encodings]

            X,y=transform(encodings)

            y=y.reshape(-1)
            output,hidden=model(X)
            loss=loss_fn(output,y)

            print_loss+=loss.item()
            running_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % print_every==0:
                print_loss_avg,current=print_loss if batch==0 else print_loss/print_every,batch*len(texts)
                print_loss=0
                logger.info(f"loss:{print_loss_avg:>7f} [{current:>5d}/{size:>5d}]")
        
        return running_loss/len(dataloader)

    def test_loop(self,dataloader,model,loss_fn):
        size=len(dataloader.dataset)
        total_loss=0
        correct=0
        sent_length=0

        model.eval()
        with torch.no_grad():
            for batch,texts in enumerate(dataloader):
                encodings=self.tokenizer.encode_batch(texts)
                encodings=[e.ids for e in encodings]

                X,y=transform(encodings)

                y=y.reshape(-1)
                mask=(y!=0)

                sent_length+=mask.type(torch.float).sum()

                output,hidden=model(X)
                total_loss+=loss_fn(output,y).item()
                correct+=torch.masked_select(output.argmax(1)==y,mask).type(torch.float).sum().item()

        total_loss/=len(dataloader)
        correct/=sent_length

        logger.info(f"Test Error:\n Accuracy:{100*correct:>0.1f}%, Avg loss:{total_loss:>8f}\n")

        return total_loss

    def generate(self,count):
        self.model.eval()
        return self.decoder.decode(self.model,self.tokenizer,count=count,max_length=100,top_p=0.6)
    
    def _save(self,dir):

        self.tokenizer.save(dir)
        torch.save(self.model,os.path.join(dir,"pytorch_model.bin"))


