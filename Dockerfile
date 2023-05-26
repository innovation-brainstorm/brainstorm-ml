FROM --platform=linux/amd64 python:3.8-slim-buster

WORKDIR /brainstorm-ml

# 升级pip
RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY src src
COPY models models

CMD ["python3", "src/app.py"]