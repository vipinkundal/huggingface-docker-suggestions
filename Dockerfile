FROM python:3.10.9

COPY ./requirements.txt /webapp/requirements.txt

WORKDIR /webapp

RUN pip install --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y libhdf5-dev libhdf5-serial-dev

RUN pip install -U sentence-transformers

RUN pip install -r requirements.txt

COPY webapp/* /webapp

# ENTRYPOINT [ "uvicorn" ]

CMD ["uvicorn", "main:app", "--host", "localhost", "--port", "5001", "--reload"]