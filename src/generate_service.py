import os
import requests

import pandas as pd
from utils.data_utils import read_csv,write_txt,write_csv,read_txt
from generator import get_generator
from generator.char_generator import CharGenerator
from generator.word_generator import WordGenerator
from schemas import CreateTaskQuery,UpdateTaskResponse,Status
from config import config
from urllib3.exceptions import MaxRetryError
from requests.exceptions import ConnectionError
import logging
logger=logging.getLogger(__name__)

def write_output(generated_text_list,generate_cols,output_path):

    output_df=pd.DataFrame(generated_text_list,columns=generate_cols)

    write_csv(output_path,output_df)
    # write_txt(output_path,generated_text_list)

def return_result(response_payload:UpdateTaskResponse):
    
    headers = {
        'Content-Type': 'application/json'
    }
    response=requests.request("POST", config.SERVICE_URL, data=response_payload.json(), headers=headers)

    res = response.json()
    logger.info(f"response from service update task: {res}")


def get_model_by_id(model_id):
    if not os.path.exists(config.GENERATE_MODEL_PATH):
        return False
    folders=os.listdir(config.GENERATE_MODEL_PATH)
    return model_id in folders


def process(query:CreateTaskQuery):

    # 1.check input: file existing, col existing
    # 2. generate training files
    # 3 decide which model to run
    # 4. train
    # 5. generate
    # 6 send result to java service
    try:

        file_path=query.filePath
        generate_cols=[query.columnName]
        count=query.expectedCount
        task_id=query.taskId

        model_dir=os.path.join(config.GENERATE_MODEL_PATH,query.modelId)
        session_dir=os.path.dirname(os.path.dirname(file_path))
        run_dir=os.path.join(os.path.dirname(file_path),query.columnName)
        output_path=os.path.join(session_dir,f"{query.columnName}.csv")

        try:
            os.mkdir(run_dir)
        except Exception as e:
            pass


        if query.usingExistModel:
            try:
                model_type=read_txt(os.path.join(model_dir,"model_type"))
                generator=get_generator(model_type[0],None,run_dir,model_dir)
            except FileNotFoundError as e:
                logger.error(f"no existing model in {model_dir}")
                generator=None

        if not query.usingExistModel or not generator:
        
            concat_pattern="<SEP>"

            df=read_csv(file_path)
            concated_data=df.loc[:,generate_cols].agg(concat_pattern.join,axis=1).to_list()
            concated_data=[data for data in concated_data if data.strip()!=""]

            word_counts=[len(row.split(" "))for row in concated_data]
            avg_word_count=sum(word_counts)/len(word_counts)

            if avg_word_count>5:
                generator=WordGenerator(concated_data,run_dir,os.path.join(config.BASE_MODEL_PATH,"distilgpt2"))
            else:
                generator=CharGenerator(concated_data,run_dir)


            generator.run()
            generator.save(model_dir)
            
        generated_text_list=generator.generate(count)

        write_output(generated_text_list,generate_cols,output_path)

        response_payload=UpdateTaskResponse(sessionId=query.sessionId,taskId=task_id,status=Status.COMPLETED,
                                    actualCount=len(generated_text_list),columnName=query.columnName,filePath=output_path)

        return_result(response_payload)
    except (ConnectionError,MaxRetryError) as e:
        logger.error("cannot connect to service",exc_info=True)
    except Exception as e:
        logger.error("generate failed",exc_info=True)
        response_payload=UpdateTaskResponse(sessionId=query.sessionId,taskId=task_id,status=Status.ERROR,
                                    actualCount=0,columnName=query.columnName,filePath="")
    
        return_result(response_payload)
    



# if __name__=="__main__":
#     #  generate_data_main("data/names/name_soeid/name_soeid.csv",["full_name"],1000)
#     query=CreateTaskQuery(filePath="data/news/news.csv",columnName="headline",ExpectedCount=1000,
#                           sessionID="",taskID="",status="NEW")
#     process(query)