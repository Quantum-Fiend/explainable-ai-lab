# Base Image with Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for gcc/xgboost sometimes)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for Jupyter or Dashboard
EXPOSE 8888 8050

# Default command: Run the extraction/inference script or open jupyter
# For production, we might run the dashboard or an API
CMD ["python", "deployment_test.py"]
