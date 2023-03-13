
import torch
from torch.utils.data import Dataset,DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments,\
                        trainer,DataCollatorForLanguageModeling,get_scheduler,pipeline


class WordGenerator(object):

    def __init__(self,model_path):
        self.model_path=model_path

        self.model=GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer=GPT2Tokenizer.from_pretrained(model_path)

        self.tokenizer.pad_token=self.tokenizer.bos_token
        self.tokenizer.add_special_tokens({'pad_token':'<pad>'})

        self.mode.resize_token_embeddings(len(self.tokenizer))


    

    def train(self,train_data:Dataset,eval_data:Dataset):

        epochs=5
        batch_size=8
        learning_rate=5e-5

        print_every=10
        max_length=200

        v=self.tokenizer.vocab_size

        train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
        eval_dataloader=DataLoader(eval_data,batch_size=batch_size,shuffle=True)

        self.model.train()
        optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)

        data_collator=DataCollatorForLanguageModeling(self.tokenizer,mlm=False,return_tensors="pt")

        train_loss=[]
        eval_loss=[]
        for i in range(epochs):
            print(f"Epoch:{i}/{epochs}..........")
            train_loss.append(self.train_loop(train_dataloader,self.model,optimizer,self.tokenizer,data_collator,max_length,print_every))
            eval_loss.append(self.test_loop(eval_dataloader,self.model,self.tokenizer,data_collator,max_length))



    def train_loop(self,dataloader,model,optimizer,tokenizer,data_collator,max_length,print_every):

        size=len(dataloader.dataset)
        print_loss=0
        running_loss=0

        for batch,texts in enumerate(dataloader):
            encodings=tokenizer(texts,truncation=True,padding="max_length",max_length=max_length)
            X=data_collator([encodings])


            output=model(**X)
            loss=output.loss

            print_loss+=loss
            running_loss+=loss



            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % print_every==0:
                print_loss_avg,current=print_loss if batch==0 else print_loss/print_every,batch*len(texts)
                print_loss=0
                print(f"loss:{print_loss_avg:>7f} [{current:>5d}/{size:>5d}]")
        
        return running_loss/len(dataloader)

    def test_loop(self,dataloader,model,tokenizer,data_collator,max_length):
        size=len(dataloader.dataset)
        total_loss=0
        correct=0
        sent_length=0

        with torch.no_grad():
            for batch,texts in enumerate(dataloader):

                encodings=tokenizer(texts,truncation=True,padding="max_length",max_length=max_length)
                X=data_collator([encodings])


                output=model(**X)
                total_loss+=output.loss

        total_loss/=len(dataloader)
        correct=0

        print(f"Test Error:\n Accuracy:{100*correct:>0.1f}%, Avg loss:{total_loss:>8f}\n")

        return total_loss

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

