FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose port for HF Spaces health check
EXPOSE 7860

# Default: run inference on hard level, 5 episodes
CMD ["python", "inference.py", "--task", "hard", "--episodes", "5"]