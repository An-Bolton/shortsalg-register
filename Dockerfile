FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "shortsalg_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
