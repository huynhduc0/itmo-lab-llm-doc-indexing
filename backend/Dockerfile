# Stage 1: Build dependencies
FROM python:3.12-slim-buster as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create final image
FROM python:3.12-slim-buster

WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Install LibreOffice (required by unstructured)
RUN apt-get update && apt-get install -y libreoffice && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8501

CMD streamlit run backend/ui.py