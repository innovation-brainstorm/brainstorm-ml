import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import transform
from tokenizers.char_tokenizer import CharacterTokenizer

from model.lstm_lm import LSTMLanguageModel

class CharGenerator(object):

    def __init__(self,model):
        self.model=model

        self.tokenizer=CharacterTokenizer()

        

    def train(self,train_data:Dataset,eval_data:Dataset):
        self.tokenizer.train(train_data.data)

        v=self.tokenizer.get_vocab_size()
        print(f"vocab count: {v}")

        hidden_size=125
        n_layers=1

        epochs=50
        batch_size=16
        learning_rate=0.0001

        print_every=10

        train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
        eval_dataloader=DataLoader(eval_data,batch_size=batch_size,shuffle=True)

        self.model=LSTMLanguageModel(v,hidden_size,v,n_layers)
        self.model.train()

        loss_fn=nn.CrossEntropyLoss(ignore_index=0)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)


        train_loss=[]
        eval_loss=[]
        for i in range(epochs):
            print(f"Epoch:{i}/{epochs}..........")
            train_loss.append(self.train_loop(train_dataloader,self.model,loss_fn,optimizer,self.tokenizer,print_every))
            eval_loss.append(self.test_loop(eval_dataloader,self.model,loss_fn,self.tokenizer))


    def train_loop(self,dataloader,model,loss_fn,optimizer,tokenizer,print_every):
        size=len(dataloader.dataset)
        print_loss=0
        running_loss=0

        for batch,texts in enumerate(dataloader):

            encodings=tokenizer.encode_batch(texts)
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
                print(f"loss:{print_loss_avg:>7f} [{current:>5d}/{size:>5d}]")
        
        return running_loss/len(dataloader)

    def test_loop(self,dataloader,model,loss_fn,tokenizer):
        size=len(dataloader.dataset)
        total_loss=0
        correct=0
        sent_length=0

        with torch.no_grad():
            for batch,texts in enumerate(dataloader):
                encodings=self.tokenizer.encode_batch(texts)
                encodings=[e.ids for e in encodings]

                X,y=transform(encodings)

                y=y.reshape(-1)
                mask=(y!=0)

                sent_length+=mask.type(torch.float).sum()

                output,hidden=self.model(X)
                total_loss+=loss_fn(output,y).item()
                correct+=torch.masked_select(output.argmax(1)==y,mask).type(torch.float).sum().item()

        total_loss/=len(dataloader)
        correct/=sent_length

        print(f"Test Error:\n Accuracy:{100*correct:>0.1f}%, Avg loss:{total_loss:>8f}\n")

        return total_loss

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

