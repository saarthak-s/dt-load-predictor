# Use the official Python 3.10 slim image to keep the runtime lightweight while still supporting a full Python toolchain.
FROM python:3.10-slim
# Set the working directory inside the container so application files are organized in one predictable place.
WORKDIR /app
# Copy only the dependency manifest first so Docker can cache it separately from the application source code.
COPY requirements.txt ./
# Install Python dependencies from the requirements file so the runtime has FastAPI, pandas, xgboost, and ML libraries available.
RUN pip install --no-cache-dir -r requirements.txt
# Copy the application source code into the container after dependency installation to maximize build cache efficiency.
COPY . .
# Expose port 8000 as the standard HTTP port for FastAPI/Uvicorn in containerized deployments.
EXPOSE 8000
# Set the default command to start Uvicorn in production mode, binding to all interfaces on port 8000.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
