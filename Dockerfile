FROM python:3.8-slim-buster

WORKDIR /brainstorm-ml

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY src src
COPY models models

CMD ["python3", "src/app.py"]