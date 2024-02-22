FROM python:3.7.0

WORKDIR /app

COPY . .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

#RUN python3 -m pip install opencv-python
RUN python3 -m  pip install --upgrade pip

RUN python3 -m pip install glcontext

RUN python3 -m pip install PyOpenGL

RUN python3 -m pip install faiss

RUN python3 -m pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:5000 server:app