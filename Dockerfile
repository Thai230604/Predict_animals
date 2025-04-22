FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime



RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

CMD ["python", "train.py"]
