FROM python:3.9
COPY ./incomeprediction /incomeprediction
WORKDIR /incomeprediction
RUN pip install -r requirements.txt
CMD ["python", "application.py"]