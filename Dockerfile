FROM python:3.7
COPY . /temp/piven
RUN pip install /temp/piven --no-cache-dir
