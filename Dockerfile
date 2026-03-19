# Use the official AWS Lambda Python base image
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.11

# Install system dependencies needed for some python packages
RUN yum install -y gcc gcc-c++ make

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Upgrade pip first so it can find pre-built wheels for all packages
RUN pip install --upgrade pip

# Install the specified packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory to the container
# This includes src/, prompts/, and handler.py
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (filename.function_name)
CMD [ "handler.handler" ]
