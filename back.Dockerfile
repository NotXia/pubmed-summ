FROM python:3.10

WORKDIR /framework
COPY framework .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

WORKDIR /web/backend
COPY web/backend .
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]