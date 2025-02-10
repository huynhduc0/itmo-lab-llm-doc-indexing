# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && ls

# Stage 2: Install LibreOffice minimally
FROM python:3.12-slim AS libreoffice_stage

RUN apt-get update && \
    apt-get install -y --no-install-recommends libreoffice-core libreoffice-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN ls -l /opt
RUN ls -l /usr/lib

# Stage 3: Create final image
FROM python:3.12-slim

WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy LibreOffice
COPY --from=libreoffice_stage /usr/lib/libreoffice /usr/lib/libreoffice
# COPY --from=libreoffice_stage /opt/libreoffice /opt/libreoffice

COPY . .
EXPOSE 8501
SHELL ["/bin/bash", "-c"]
CMD ls && source venv/bin/activate && python -m streamlit run backend/ui.py