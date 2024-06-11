FROM tiangolo/meinheld-gunicorn-flask:python3.8

# Install curl for healthchecks
RUN apt install curl

WORKDIR /app

# Setup a nonroot user for security
RUN useradd -ms /bin/bash nonroot
USER nonroot

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir --upgrade -r requirements.txt

# Copy the app
COPY . .

USER root
RUN chown -R nonroot:nonroot /app
USER nonroot

# Expose the app's port
EXPOSE 8000

# Run the FastAPI server
ENTRYPOINT ["python", "/app/main.py"]
