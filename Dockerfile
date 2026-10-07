FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY app.py server.py ./
# Pre-download the CLIP models into the image so cold starts stay fast.
RUN python -c "from app import get_clip; get_clip()"

ENV PORT=8791
EXPOSE 8791
CMD ["python", "server.py"]
