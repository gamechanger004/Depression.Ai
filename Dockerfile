# Use the official Python image from the Docker Hub
FROM python:3.12.2

# Set the working directory in the container
WORKDIR /depression.ai

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define the environment variable for Flask
ENV FLASK_APP=server.py

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
