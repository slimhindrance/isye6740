# Use a lightweight official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only necessary files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set an environment variable for the output directory (overridable)
ENV OUTPUT_DIR="/app/threads"

# Create the output directory inside the container
RUN mkdir -p $OUTPUT_DIR

# Expose the output directory as a volume (optional but useful)
VOLUME ["/app/threads"]

# Ensure the script runs with the latest environment variables
CMD ["python", "ed_download.py"]
