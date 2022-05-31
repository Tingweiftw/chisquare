FROM python:3.10.4
WORKDIR /code
COPY requirements.txt /code
RUN apt-get update -y && \
    apt-get install -y python-numpy\
    python-numpy python-scipy &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
COPY . /code
CMD python app.py