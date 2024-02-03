# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the Docker container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Command to run on container start
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501"]