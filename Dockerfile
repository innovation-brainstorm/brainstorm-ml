FROM python:3.8-slim-buster

WORKDIR /brainstorm-ml

# 升级pip
# RUN pip3 install --upgrade pip
# 创建或者修改 pip 的配置文件
RUN mkdir -p /root/.pip && echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /root/.pip/pip.conf
COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

COPY src src
COPY models models

CMD ["python3", "src/app.py"]