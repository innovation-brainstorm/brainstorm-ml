
import sys
from os import path
import random

sys.path.append(path.join(path.dirname(path.abspath(__file__)),".."))
from utils.data_utils import read_txt,write_txt,split_data


def mock_soeid_by_name():
    path="../data/names_soeid"
    first_names=read_txt(path.join(path,"first_name.txt"))
    last_names=read_txt(path.join(path,"last_name.txt"))

    full_data=[]
    k=5000
    rnd_last_names=random.choices(last_names,k=k)
    rnd_first_names=random.choices(first_names,k=k)
    
    for i,(f,l) in enumerate(zip(rnd_first_names,rnd_last_names)):

        soeid=f[0]+l[0]+"".join(str(random.randint(0,9)) for i in range(5))

        sample=f"{l}, {f} ({soeid})"

        full_data.append(sample)

    write_txt(path.join(path,"names.txt"),full_data)

    split_data(path.join(path,"names.txt"),0.7,0.1,0.2)



if __name__=="__main__":
    pass