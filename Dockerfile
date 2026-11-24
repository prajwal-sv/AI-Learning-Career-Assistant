# Start from a small Linux image that already has Python (same version as ours)
FROM python:3.14-slim

# All work happens inside the /app folder
WORKDIR /app

# Do not write .pyc files, and show logs straight away
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install the packages first (Docker remembers this step, so rebuilds are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code, data and model (the .dockerignore file removes the rest)
COPY . .

# The app listens on port 8501
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]