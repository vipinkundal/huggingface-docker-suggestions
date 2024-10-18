FROM python:3.10.9

EXPOSE 8001

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y libhdf5-dev libhdf5-serial-dev

RUN pip install -U sentence-transformers

RUN pip install -r requirements.txt

COPY app/* /app

# ENTRYPOINT [ "uvicorn" ]

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]