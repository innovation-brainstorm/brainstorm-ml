
import pathlib
import os
import torch
from torch.utils.data import Dataset,DataLoader
from generator.base_generator import BaseGenerator

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments,\
                        trainer,DataCollatorForLanguageModeling,get_scheduler,pipeline
from decoding.gpt_decoding import GPTDecoding
from dataset.text_dataset import TextDataset
from utils.early_stop import EarlyStop
from utils.data_utils import write_txt

import logging
logger=logging.getLogger(__name__)

class WordGenerator(BaseGenerator):

    model_type="WORD"
    
    epochs=5
    batch_size=8
    learning_rate=5e-5

    max_length=200

    def __init__(self,data_or_filepath,data_dir,model_path):

        self.data_or_filepath=data_or_filepath
        self.data_dir=data_dir

        self.model_path=model_path

        self.model=GPT2LMHeadModel.from_pretrained(self.model_path)
        self.tokenizer=GPT2Tokenizer.from_pretrained(self.model_path)
        self.decoder=GPTDecoding()

    def build_dataset(self):

        train_dataset=TextDataset(root=self.data_dir,split="train",max_count=200,stop_token="<|endoftext|>")
        eval_dataset=TextDataset(root=self.data_dir,split="eval",max_count=50,stop_token="<|endoftext|>")

        return train_dataset,eval_dataset


    def train_tokenizer(self,train_data:Dataset):

        self.tokenizer.pad_token=self.tokenizer.bos_token
        self.tokenizer.add_special_tokens({'pad_token':'<pad>','sep_token':"<SEP>"})
        self.model.resize_token_embeddings(len(self.tokenizer))


        v=self.tokenizer.vocab_size
        logger.info(f"vocab count: {v}")
    

    def train(self,train_data:Dataset,eval_data:Dataset):


        v=self.tokenizer.vocab_size

        train_dataloader=DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
        eval_dataloader=DataLoader(eval_data,batch_size=self.batch_size,shuffle=True)

        optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

        data_collator=DataCollatorForLanguageModeling(self.tokenizer,mlm=False,return_tensors="pt")

        early_stop=EarlyStop(patience=2,delta=0)

        train_loss_list=[]
        val_loss_list=[]
        for i in range(self.epochs):
            logger.info(f"Epoch:{i}/{self.epochs}..........")

            train_loss=self.train_loop(train_dataloader,self.model,optimizer,data_collator)
            val_loss=self.test_loop(eval_dataloader,self.model,self.tokenizer,data_collator)

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            early_stop(val_loss)
            if early_stop.early_stop:
                logger.info(f"Early Stop training")
                break
        


    def train_loop(self,dataloader,model,optimizer,data_collator):

        print_every=10

        size=len(dataloader.dataset)
        print_loss=0
        running_loss=0

        model.train()

        for batch,texts in enumerate(dataloader):
            encodings=self.tokenizer(texts,truncation=True,padding="max_length",max_length=self.max_length)
            
            X=data_collator([encodings])


            output=model(**X)
            loss=output.loss

            print_loss+=loss
            running_loss+=loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % print_every==0:
                print_loss_avg,current=print_loss if batch==0 else print_loss/print_every,batch*len(texts)
                print_loss=0
                logger.info(f"loss:{print_loss_avg:>7f} [{current:>5d}/{size:>5d}]")
        
        return running_loss/len(dataloader)

    def test_loop(self,dataloader,model,tokenizer,data_collator):
        size=len(dataloader.dataset)
        total_loss=0
        correct=0
        sent_length=0

        model.eval()
        with torch.no_grad():
            for batch,texts in enumerate(dataloader):

                encodings=tokenizer(texts,truncation=True,padding="max_length",max_length=self.max_length)
                X=data_collator([encodings])


                output=model(**X)
                total_loss+=output.loss

        total_loss/=len(dataloader)
        correct=0

        logger.info(f"Test Error:\n Accuracy:{100*correct:>0.1f}%, Avg loss:{total_loss:>8f}\n")

        return total_loss
    
    def generate(self,count):
        self.model.eval()
        return self.decoder.decode(self.model,self.tokenizer,count=count,max_length=100)
    
    def _save(self,dir):
        self.tokenizer.save_pretrained(dir)
        self.model.save_pretrained(dir)





