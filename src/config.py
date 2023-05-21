import os


class DefaultConfig(object):
    PORT=os.environ.get("PORT", default=8000)
    SERVICE_URL=os.environ.get("SERVICE_URL", default="http://localhost:8080/api/task/updateStatus")
    BASE_MODEL_PATH=os.environ.get("BASE_MODEL_PATH", default="models")


config=DefaultConfig()