FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get -y install \
    bash ffmpeg wget unzip git

# Setup working directory
WORKDIR /rife

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Make sure model weights are available
RUN mkdir -p train_log && \
    if [ ! -f "train_log/flownet.pkl" ]; then \
        wget -O train_log/flownet.pkl https://github.com/hzwer/Practical-RIFE/releases/download/1.0/flownet.pkl; \
    fi

# Add CLI tools
ADD docker/inference_img /usr/local/bin/inference_img
RUN chmod +x /usr/local/bin/inference_img
ADD docker/inference_video /usr/local/bin/inference_video
RUN chmod +x /usr/local/bin/inference_video

# Expose port for potential web interface
EXPOSE 8080

# Set working directory for mounting host volumes
WORKDIR /host

# Define entrypoint
ENTRYPOINT ["bash"]
ENTRYPOINT ["/bin/bash"]

ENV NVIDIA_DRIVER_CAPABILITIES all