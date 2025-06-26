FROM python:3.10-slim

WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your Flask app code
COPY index.py .

# Ensure stdout logs show up immediately
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=development

# Run the Flask app
CMD ["python", "index.py"]
