FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Note: this container runs the app itself. It expects an Ollama server
# reachable at OLLAMA_HOST (see docker-compose.yml, which runs Ollama as a
# separate service on your machine so the model weights aren't rebuilt
# into this image).
ENV OLLAMA_HOST=http://ollama:11434

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
