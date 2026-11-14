# On-premise CPU image: no cloud account, no GPU, no outbound traffic at run time.
# Not built or run during local preparation (Docker was unavailable); see README.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY config ./config
VOLUME ["/data", "/output"]
ENTRYPOINT ["python"]
CMD ["count_flow.py", "--input", "/data/clip.mp4", "--config", "/data/config.json", "--detector", "onnx:/data/model.onnx", "--output", "/output"]
