# Use a slim Python 3.11 image for a modern and lightweight footprint
FROM python:3.11-slim-bookworm

# Set the working directory
WORKDIR /app

# Make pip more resilient during large dependency downloads in Docker builds.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_PROGRESS_BAR=off

# Install system dependencies
# Bookworm is the newer Debian stable, offering better security and package support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
# This prevents reinstalling all packages when you change source code
COPY requirements.txt .

# Upgrade packaging tools first, then install dependencies with a longer timeout.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --retries 10 --timeout 300 -r requirements.txt

# Copy the rest of the source code
COPY . .

# Set environment variables for better Python behavior in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the Flask port
EXPOSE 5000

# Command to run the application
# Note: Use Gunicorn for production instead of 'python app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
