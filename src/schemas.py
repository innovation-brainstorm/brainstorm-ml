from enum import Enum
from pydantic import BaseModel


class Status(Enum):
    NEW="NEW"
    RUNNING="RUNNING"
    COMPLETED="COMPLETED"
    ERROR="ERROR"



class CreateTaskQuery(BaseModel):

    sessionId:str
    taskId:str
    columnName:str
    expectedCount:int
    status:Status="NEW"
    filePath:str

class CreateTaskResponse(BaseModel):
    sessionId:str=""
    taskId:str=""
    status:Status="RUNNING"

class UpdateTaskResponse(BaseModel):
    sessionId:str
    taskId:str
    columnName:str
    actualCount:int
    status:Status
    filePath:str
