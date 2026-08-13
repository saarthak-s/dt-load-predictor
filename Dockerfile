# Use the official Python 3.10 slim image to keep the runtime lightweight while still supporting a full Python toolchain.
FROM python:3.10-slim
# Set the working directory inside the container so application files are organized in one predictable place.
WORKDIR /app
# Copy the uv project manifest first so Docker can cache dependency resolution and installation efficiently.
COPY pyproject.toml ./
# Install uv inside the container so Python dependencies can be managed with the uv-native workflow.
RUN pip install --no-cache-dir uv
# Create a virtual environment inside the container and install the project dependencies from pyproject.toml.
RUN uv venv && uv pip install --python .venv/bin/python .
# Copy the application source code into the container after dependency installation to maximize build cache efficiency.
COPY . .
# Expose port 8000 as the standard HTTP port for FastAPI/Uvicorn in containerized deployments.
EXPOSE 8000
# Set the default command to start Uvicorn in production mode, binding to all interfaces on port 8000.
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
