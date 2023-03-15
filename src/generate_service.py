import os
from utils.data_utils import split_data,read_csv
from generator.char_generator import CharGenerator
from generator.word_generator import WordGenerator

def process():
    # argument: task_id, file_path, generate_col, meta_col, count
    # return status for receive task
    # 1. add task to queue
    # 2. check file existing
    # 2. return status for receive task

    pass


def generate_data_main(file_path,generate_cols,count):
    # 1.check input: file existing, col existing
    # 2. generate training files
    # 3 decide which model to run

    # 4. train
    # 5. generate
    # 6 send result to java service
    task_dir=os.path.dirname(file_path)
    run_dir=os.path.join(task_dir,"run")
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

    # split_data(concated_data,run_dir,0.8,0.2,0)

    generator.run()
    generator.generate(10)


if __name__=="__main__":
     generate_data_main("data/names/name_soeid/name_soeid.csv",["full_name"],1000)