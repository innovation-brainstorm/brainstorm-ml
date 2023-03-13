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
            f.write(row)
            f.write("\n")


def read_csv(path,orient="list"):

    df=pd.read_csv(path,encoding="utf-8")
    return df.to_dict(orient=orient)


def write_csv(path,df):
    df.to_csv(path)


def remove_non_ascii_2(string):
    return string.encode("ascii",errors="ignore").decode()

def split_data(file_path,train_ratio,eval_ratio,test_ratio,criteria=None):
    assert train_ratio+eval_ratio+test_ratio==1

    ext=file_path.split(".")[-1]

    if ext=="txt":
        data=read_txt(file_path)
        write_func=write_txt

    elif ext=="csv":
        data=read_csv(file_path)
        write_func=write_csv
    else:
        print("no such ext")
        return 

    filtered_data=[]
    for row in data:
        non_ascii_text=remove_non_ascii_2(row).strip()
        if non_ascii_text!="" and (criteria is None or criteria(non_ascii_text)):
            filtered_data.append(non_ascii_text)
    
    n=len(filtered_data)
    print(filtered_data[:5])

    x_train,x_other=train_test_split(filtered_data,train_size=train_ratio,random_state=42,shuffle=True)
    x_eval,x_test=train_test_split(x_other,train_size=eval_ratio/(eval_ratio+test_ratio))

    print(f"total count: {n}, train count: {len(x_train)}, eval count: {len(x_eval)}, test count: {len(x_test)}")

    dir_path=os.path.dirname(file_path)

    write_func(os.path.join(dir_path,"train."+ext),x_train)
    write_func(os.path.join(dir_path,"eval."+ext),x_train)
    write_func(os.path.join(dir_path,"test."+ext),x_train)


# TODO: conver to datacollator
def transform(encodings):

    encodings_tensor=torch.longTensor(encodings)

    input_tensor=encodings_tensor[:,:-1]
    output_tensor=encodings_tensor[:,1:]

    return input_tensor,output_tensor