# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:4.10.3

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

# Create the Conda environment and install cmdstan
RUN conda create -n stan -c conda-forge cmdstan
RUN echo "source activate stan" > ~/.bashrc
ENV PATH /opt/conda/envs/stan/bin:$PATH

# Run setup.py
RUN python setup.py install

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME onnx-stan

# Run the application when the container launches
CMD ["python", "your_main_script.py"]
