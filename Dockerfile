# Multi-stage Docker build for Mesa LLM Assistant

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY mesa_llm/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY mesa_llm/ ./mesa_llm/
COPY setup.py .

# Install the application
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mesa_user
RUN chown -R mesa_user:mesa_user /app
USER mesa_user

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["python", "-m", "mesa_llm.main"]