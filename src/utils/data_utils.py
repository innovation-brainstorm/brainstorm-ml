import os
import csv
import json
import unicodedata
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def read_txt(path):
    data=[]

    with open(path,"r",encoding="utf-8") as f:
        for row in f:
            data.append(row.strip())
    return data


def write_txt(path,data):
    with open(path,"w",encoding="utf-8") as f:
        for row in data:
            row=row.replace("\r\n","\n").replace("\n","\\n")
            f.write(row)
            f.write("\n")


def read_csv(path,orient="list"):

    df=pd.read_csv(path,encoding="utf-8",index_col=False,dtype=str,keep_default_na=False)
    return df


def write_csv(path,df):
    df.to_csv(path,index=False)


def remove_non_ascii_2(string):
    return string.encode("ascii",errors="ignore").decode()

def split_data(data_or_filepath,output_dir,train_ratio,eval_ratio,test_ratio,criteria=None):
    assert train_ratio+eval_ratio+test_ratio==1

    if type(data_or_filepath)==list:
        data=data_or_filepath

    elif type(data_or_filepath)==str:

        data=read_txt(data_or_filepath)


    filtered_data=[]
    for row in data:
        non_ascii_text=remove_non_ascii_2(row).strip()
        if non_ascii_text!="" and (criteria is None or criteria(non_ascii_text)):
            filtered_data.append(non_ascii_text)
    
    n=len(filtered_data)
    print("example data:\n",filtered_data[:5])

    x_train,x_other=train_test_split(filtered_data,train_size=train_ratio,random_state=42,shuffle=True)
    if test_ratio>0:
        x_eval,x_test=train_test_split(x_other,train_size=eval_ratio/(eval_ratio+test_ratio))
    else:
        x_eval=x_other
        x_test=[]

    print(f"total data count: {n}, train count: {len(x_train)}, eval count: {len(x_eval)}, test count: {len(x_test)}")


    write_txt(os.path.join(output_dir,"train.txt"),x_train)
    write_txt(os.path.join(output_dir,"eval.txt"),x_eval)
    write_txt(os.path.join(output_dir,"test.txt"),x_test)


# TODO: conver to datacollator
def transform(encodings):

    encodings_tensor=torch.LongTensor(encodings)

    input_tensor=encodings_tensor[:,:-1]
    output_tensor=encodings_tensor[:,1:]

    return input_tensor,output_tensor