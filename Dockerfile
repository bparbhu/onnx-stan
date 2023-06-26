# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container to /onnx-stan
WORKDIR /onnx-stan

# Add the current directory contents into the container at /onnx-stan
ADD . /onnx-stan

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install other necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    libboost-all-dev

# Run setup.py
RUN python setup.py install

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME onnx-stan

# Run the application when the container launches
CMD ["python", "your_main_script.py"]
