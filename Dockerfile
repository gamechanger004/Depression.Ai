# Use the official Python image from the Docker Hub
FROM python:3.12-alpine

RUN apk add --no-cache build-base python3-dev py3-pip

# Set the working directory in the container
WORKDIR /app

# Copy the application code and requirements file into the container
COPY . .



# Install the Python dependencies specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Expose the port that the Flask app will run on
EXPOSE 8080

# Command to run the Flask application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "server:app"]
