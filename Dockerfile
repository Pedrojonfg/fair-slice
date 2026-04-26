FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["streamlit", "run", "src/fair-slice/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
