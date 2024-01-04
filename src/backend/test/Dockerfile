FROM python:3.10-slim

WORKDIR /app

# Install Poetry
RUN apt-get update && apt-get install gcc g++ curl build-essential postgresql-server-dev-all -y
RUN apt-get update && apt-get install procps -y
# opencv
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1
RUN python -m pip install --upgrade pip  --force -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install shapely==2.0.1 --force -i https://mirrors.aliyun.com/pypi/simple/


RUN python -m pip install pipx --upgrade --force -i https://mirrors.aliyun.com/pypi/simple/

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
#RUN python -m pipx install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN pip install uvicorn
RUN pip install pandas
RUN pip install pydantic
RUN pip install minio==7.2.0
# docker build -t python-poetry:1.0 ./