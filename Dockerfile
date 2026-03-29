FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY . .

# Pre-build outputs dir so it exists even before training
RUN mkdir -p outputs

# Default: run the FastAPI server.
# Override CMD to run the dashboard or training script instead.
EXPOSE 8000
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
