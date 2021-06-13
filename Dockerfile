FROM python:3.8.7-slim-buster
LABEL org.opencontainers.image.source https://github.com/chelnak/numerai-submission-workflow
ADD . /app
WORKDIR /app
RUN apt-get -y update && apt-get install -y libgomp1
RUN pip install -e ./lib/modlr
RUN pip install -r ./requirements.txt
ENTRYPOINT ["python", "submit.py"]
