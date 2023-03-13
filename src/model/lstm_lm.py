import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers):
        super(LSTMLanguageModel,self).__init__()
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