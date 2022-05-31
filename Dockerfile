FROM python:3.10

RUN apt-get update -y && \
    apt-get install -y python3-numpy\
    python3-numpy python3-scipy &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt /code
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
COPY . /code
CMD python app.py