# Use the official Python image from the Docker Hub
FROM python:3.12-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the application code and requirements file into the container
COPY . .

# Install system dependencies (if required)
# For example, you might need `libgl1` for certain libraries like OpenCV
RUN apk update && apk add --no-cache libgl1

# Install the Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Expose the port that the Flask app will run on
EXPOSE 8080

# Command to run the Flask application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
