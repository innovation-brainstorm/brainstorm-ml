import os
from utils.data_utils import read_txt,read_csv

from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self,root,split="train",max_count=None,ext="txt",stop_token="",start_token=""):
        self.root=root
        self.split=split
        self.ext=ext
        self.stop_token=stop_token
        self.start_token=start_token

        self.data=self._load_data(max_count)


    def _load_data(self,max_count=None):
        file_path=os.path.join(self.root,f"{self.split}.{self.ext}")

        if self.ext=="txt":
            data=read_txt(file_path)
        elif self.ext=="csv":
            data=read_csv(file_path)

        else:
            print("wrong ext type")
            return []

        if max_count:
            data=data[:max_count]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        sample=self.start_token+self.data[idx].strip()+self.stop_token

        return sample

    

