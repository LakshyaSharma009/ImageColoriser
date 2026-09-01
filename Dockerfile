# Optional container image for the Streamlit UI (and, via docker-compose, the
# FastAPI service). Docker is never required to run this project -- `python
# main.py` and `streamlit run streamlit_app.py` continue to work directly.
FROM python:3.11-slim

# OpenCV's Python wheels need these system libraries for image codecs even
# though everything else is installed via pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV IMAGECOLORIZER_MODEL_DIR=/app/Model \
    IMAGECOLORIZER_HISTORY_DIR=/app/history \
    IMAGECOLORIZER_LOG_DIR=/app/logs \
    IMAGECOLORIZER_EVALUATION_DIR=/app/evaluation

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
