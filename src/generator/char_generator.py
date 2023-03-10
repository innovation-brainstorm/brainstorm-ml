import torch
import torch.nn as nn

from utils.data_utils import transform

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers):
        super(LSTM,self).__init__()
        self.embedding=nn.Embedding(input_size,hidden_size,0)
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.lstm=nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.dropout=nn.Dropout(0.3)

        self.fc=nn.Linear(hidden_size,output_size)

    
    def forward(self,input_seq,hidden=None,cell=None):

        if hidden is None:
            hidden=self.init_hidden(len(input_seq))
        if cell is None:
            cell=self.init_hidden(len(input_seq))
        
        output=self.embedding(input_seq)

        output,hidden=self.lstm(output,(hidden,cell))

        output=output.contiguous().view(-1,self.hidden_size)
        output=self.dropout(output)
        output=self.fc(output)

        return output, (hidden,cell)

    def init_hidden(self,batch_size):
        return torch.zeros(self.n_layers,batch_size,self.hidden_size)



class CharGenerator(object):

    def __init__(self,model):
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer

    def train(self,dataloader,tokenizer):
        size=len(dataloader.dataset)
        print_loss=0
        running_loss=0

        for batch,texts in enumerate(dataloader):

            encodings=tokenizer.encode_batch(texts)
            encodings=[e.ids for e in encodings]

            X,y=transform(encodings)

            y=y.reshape(-1)
            output,hidden=self.model(X)
            loss=self.loss_fn(output,y)

            print_loss+=loss.item()
            running_loss+=loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % print_every==0:
                print_loss_avg,current=print_loss if batch==0 else print_loss/print_every,batch*len(texts)
                print_loss=0
                print(f"loss:{print_loss_avg:>7f} [{current:>5d}/{size:>5d}]")
        
        return running_loss/len(dataloder)

    def test(self,dataloader):
        size=len(dataloader.dataset)
        total_loss=0
        correct=0
        sent_length=0

        with torch.no_grad():
            for batch,texts in enumerate(dataloader):
                encodings=tokenizer.encode_batch(texts)
                encodings=[e.ids for e in encodings]

                X,y=transform(encodings)

                y=y.reshape(-1)
                mask=(y!=0)

                sent_length+=mask.type(torch.float).sum()

                output,hidden=model(X)
                total_loss+=loss_fn(output,y).item()
                correct+=torch.masked_select(output.argmax(1)==y,mask).type(torch.float).sum().item()

        total_loss/=len(dataloder)
        correct/=sent_length

        print(f"Test Error:\n Accuracy:{100*correct:>0.1f}%, Avg loss:{total_loss:>8f}\n")

        return total_loss

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

