FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Train the demo model and save its artifact into the image.
RUN python src/train.py

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
