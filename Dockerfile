FROM python:3.10.4
WORKDIR /code
COPY requirements.txt /code
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
COPY . /code
CMD python app.py