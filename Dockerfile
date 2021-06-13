FROM python:3.8.7-slim-buster
ADD . /app
WORKDIR /app
RUN pip install -e ./lib/modlr
RUN pip install -r ./requirements.txt
ENTRYPOINT ["python", "submit.py"]
