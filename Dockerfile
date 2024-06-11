# Use the official Python image from the Docker Hub
FROM python:3-alpine3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY . /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Install the dependencies specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Expose the port that the Flask app will run on
EXPOSE 8080

# Command to run the Flask application
CMD ["gunicorn","--blind", "0.0.0.0:8080","app:app"]
