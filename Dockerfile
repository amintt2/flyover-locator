FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY dopplergeo ./dopplergeo
RUN pip install --no-cache-dir ".[web]" "yt-dlp[default]"

ENV DATA_DIR=/data \
    YTDLP_ARGS="--js-runtimes node" \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fs http://localhost:8000/api/config || exit 1

CMD ["uvicorn", "dopplergeo.web:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
