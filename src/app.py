from flask import Flask,request,jsonify
from concurrent.futures import ThreadPoolExecutor
from schemas import CreateTaskQuery,CreateTaskResponse,Status

from generate_service import process

app=Flask(__name__)
executor=ThreadPoolExecutor(max_workers=1)



@app.route("/task/createTask",methods=["POST"])
def create_task():
    # sessionID,taskID,columnName,ExceptedCount,status,filePath
    # sessionID, taskID, status
    
    try:
        query=CreateTaskQuery(**request.json)

        response=CreateTaskResponse(sessionId=query.sessionId,
                            taskId=query.taskId,status=Status.RUNNING)

        future=executor.submit(process,query)


    except Exception as e:
        print("create task error:",str(e))
        raise
        
    return app.response_class(
                response=response.json(),
                mimetype='application/json'
        )

@app.route("/")
def hello():
    return "Hello a!"

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"message":"internel error"}),500


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=False,port=8000)