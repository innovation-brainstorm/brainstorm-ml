import json
from flask import Flask,request,jsonify
from concurrent.futures import ThreadPoolExecutor
from schemas import CreateTaskQuery,CreateTaskResponse,Status

from generate_service import process,get_model_by_id
from config import config
import logging

log_format = "%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d::%(message)s"
logging.basicConfig(level='INFO',format=log_format)
logger=logging.getLogger(__name__)

app=Flask(__name__)

executor=ThreadPoolExecutor(max_workers=1)



@app.route("/task/createTask",methods=["POST"])
def create_task():
    # sessionID,taskID,columnName,ExceptedCount,status,filePath
    # sessionID, taskID, status
    
    try:
        logger.info(request.json)
        query=CreateTaskQuery(**request.json)

        response=CreateTaskResponse(sessionId=query.sessionId,
                            taskId=query.taskId,status=Status.RUNNING)

        future=executor.submit(process,query)


    except Exception as e:
        logger.error("create task error:", exc_info=True)
        raise
        
    return app.response_class(
                response=response.json(),
                mimetype='application/json'
        )

@app.route("/task/model/<model_id>",methods=["GET"])
def get_model(model_id):
    try:
        response=get_model_by_id(model_id)    
    except Exception as e:
        logger.error("get model error:",exc_info=True)
        response=False
    
    return app.response_class(
                response=json.dumps(response),
                mimetype='application/json'
        )

@app.route("/")
def hello():
    return "Hello a!"

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"message":"internel error"}),500


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=False,port=config.PORT)