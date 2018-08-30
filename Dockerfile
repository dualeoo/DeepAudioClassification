FROM tensorflow/tensorflow:1.6.0
MAINTAINER Guang Yang <garry.yangguang@gmail.com>

RUN apt-get update -y && apt-get install -y \
    libmagic-dev \
    sox \
    libsox-fmt-mp3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
