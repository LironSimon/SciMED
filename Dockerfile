# How to use it locally:
#   docker build -t scimed .
#   docker run scimed

FROM python:3.10-slim-bullseye

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "./main.py"] 


