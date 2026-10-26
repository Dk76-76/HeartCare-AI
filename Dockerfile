# Base image
FROM python:3.10-slim

# Working directory set kar rahe hain
WORKDIR /app

# Requirements file copy karke install kar rahe hain
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saari files copy kar rahe hain
COPY . .

# FastAPI server ko run karne ka command
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"