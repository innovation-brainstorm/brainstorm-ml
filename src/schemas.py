from enum import Enum
from pydantic import BaseModel


class Status(Enum):
    NEW="NEW"
    RUNNING="RUNNING"
    COMPLETED="COMPLETED"
    ERROR="ERROR"



class CreateTaskQuery(BaseModel):

    sessionID:str
    taskID:str
    columnName:str
    expectedCount:int
    status:Status="NEW"
    filePath:str

class CreateTaskResponse(BaseModel):
    sessionID:str=""
    taskID:str=""
    status:Status="RUNNING"

class UpdateTaskResponse(BaseModel):
    sessionID:str
    taskID:str
    columnName:str
    actualCount:int
    status:Status
    filePath:str
