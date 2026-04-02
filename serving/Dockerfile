FROM python:3.11-slim

WORKDIR /app

COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY serving/ /app/serving/

ENV PYTHONPATH=/app
ENV GRADING_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
