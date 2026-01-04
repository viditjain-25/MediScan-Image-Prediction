# Use Python 3.10.11 slim image
FROM python:3.10.11-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose port your app runs on (change if needed)
EXPOSE 8000

# Command to run your app (change app.py to your main file)
CMD ["python", "app.py"]
