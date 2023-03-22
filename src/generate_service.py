import os
import http.client

import pandas as pd
from utils.data_utils import read_csv,write_txt,write_csv
from generator.char_generator import CharGenerator
from generator.word_generator import WordGenerator
from schemas import CreateTaskQuery,UpdateTaskResponse,Status
from settings import SERVICE_URL,SERVICE_PORT


def write_output(generated_text_list,generate_cols,output_path):

    output_df=pd.DataFrame(generated_text_list,columns=generate_cols)

    write_csv(output_path,output_df)
    # write_txt(output_path,generated_text_list)

def return_result(response_payload:UpdateTaskResponse):
    conn = http.client.HTTPSConnection(SERVICE_URL, SERVICE_PORT)
    
    headers = {
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/task/updateStatus", response_payload.json(), headers)

    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))


def process(query:CreateTaskQuery):

    # 1.check input: file existing, col existing
    # 2. generate training files
    # 3 decide which model to run
    # 4. train
    # 5. generate
    # 6 send result to java service

    file_path=query.filePath
    generate_cols=[query.columnName]
    count=query.ExpectedCount
    task_id=query.taskID


    task_dir=os.path.dirname(file_path)
    run_dir=os.path.join(task_dir,"run_"+task_id)
    output_path=os.path.join(run_dir,"output.csv")

    try:
        os.mkdir(run_dir)
    except Exception as e:
        pass
    
    concat_pattern="[SEP]"

    df=read_csv(file_path)
    concated_data=df.loc[:,generate_cols].agg(concat_pattern.join,axis=1).to_list()

    word_counts=[len(row.split(" "))for row in concated_data]
    avg_word_count=sum(word_counts)/len(word_counts)

    if avg_word_count>5:
        generator=WordGenerator(concated_data,run_dir)
    else:
        generator=CharGenerator(concated_data,run_dir)


    generator.run()
    generated_text_list=generator.generate(count)

    write_output(generated_text_list,generate_cols,output_path)

    response_payload=UpdateTaskResponse(sessionID=query.sessionID,taskID=task_id,status=Status.COMPLETED,
                                actualCount=len(generated_text_list),columnName=query.columnName,filePath=output_path)

    return_result(response_payload)



# if __name__=="__main__":
#     #  generate_data_main("data/names/name_soeid/name_soeid.csv",["full_name"],1000)
#     query=CreateTaskQuery(filePath="data/news/news.csv",columnName="headline",ExpectedCount=1000,
#                           sessionID="",taskID="",status="NEW")
#     process(query)