FROM tiangolo/meinheld-gunicorn-flask:python3.8

LABEL maintainer="Agile Scientific"

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app
