import pathlib
import os
from utils.data_utils import write_txt,read_txt
from generator.base_generator import BaseGenerator
import subprocess
import logging
logger=logging.getLogger(__name__)


class CustomGenerator(BaseGenerator):

    def __init__(self,data_or_filepath,data_dir,model_path):

        self.data_or_filepath=data_or_filepath
        self.data_dir=data_dir
        self.model_path=model_path


    def run(self):
        pass
        
   
    def train(self):
        pass

    def generate(self,count):
        model_entry_path=pathlib.Path(self.model_path,"main.sh")
        output_file_path=pathlib.Path(self.data_dir,"output.txt")
        p=subprocess.Popen(f"bash {model_entry_path} {self.data_or_filepath} {output_file_path} {count}",stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
        p_stderr=p.stderr.readlines()
        if p_stderr:
            err_msg=".".join(msg.decode() for msg in p_stderr)
            logger.error(f"run custom model failed. model path: {model_entry_path}. error: {err_msg}")
            raise Exception("custom model failed")
        p_stdout=p.stdout.readlines()
        print(p_stdout)

        output_texts=read_txt(output_file_path)

        return output_texts

    def save(self,dir):
        pass





    