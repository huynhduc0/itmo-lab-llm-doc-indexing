FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "ui.py", "--server.port=8000", "--server.address=0.0.0.0"]
