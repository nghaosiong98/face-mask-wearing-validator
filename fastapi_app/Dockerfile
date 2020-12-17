FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

