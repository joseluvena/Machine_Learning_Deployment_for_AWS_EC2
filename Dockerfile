FROM python:3.9.7

WORKDIR /mba_admission_app

COPY requirements.txt .
COPY model_pickle .

RUN pip install -r requirements.txt

COPY ./app ./app

EXPOSE 5000

CMD ["python", "./app/app.py"]