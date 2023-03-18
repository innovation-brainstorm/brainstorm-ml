import os
import random
import unicodedata
import pandas as pd
from unidecode import unidecode

print(__file__)

def read_txt(path):
    data=[]

    with open(path,"r",encoding="utf-8") as f:
        for row in f:
            data.append(row.strip())
    return  data
    
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




def generate_soeid(output_path,last_name_path,first_name_path,k):
    first_names=read_txt(first_name_path)
    last_names=read_txt(last_name_path)

    full_data=[]

    rnd_last_names=random.choices(last_names,k=k)
    rnd_first_names=random.choices(first_names,k=k)
    for i,(f,l) in enumerate(zip(rnd_first_names,rnd_last_names)):
        f=unidecode(f).replace(" ","")
        l=unidecode(l).strip()

        soeid=f[0]+l[0]+"".join(str(random.randint(0,9)) for i in range(5))
        sample=f"{l}, {f} ({soeid})"

        full_data.append(sample)
    
    df=pd.DataFrame(full_data,columns=["full_name"])
    write_csv(output_path,df)


def generate_news(input_path,output_path,k=5000):

    df=pd.read_json(input_path)
    df=df.sample(k)
    write_csv(output_path,df)

if __name__=="__main__":
    # generate_soeid("data/names/name_soeid/name_soeid.csv","data/names/last_name_ch.txt","data/names/first_name_ch.txt",k=5000)
    generate_news("data/news/News_Category_Dataset_v3.json","data/news/news.csv")