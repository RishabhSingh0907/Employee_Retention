# Use the official Python image from Docker Hub
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Specify the command to run your application
CMD ["python", "./main_script.py"]
