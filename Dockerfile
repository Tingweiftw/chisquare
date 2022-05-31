FROM arm32v7/python:3.10-buster
RUN apt-get -y install libblas-dev
RUN apt-get -y install gfortran
RUN apt-get -y install libopenblas-base
RUN apt-get -y install libatlas-base-dev
WORKDIR /code
COPY requirements.txt /code
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
COPY . /code
CMD python app.py