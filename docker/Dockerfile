FROM nvcr.io/nvidia/pytorch:21.05-py3

ENV TZ=Europe/London

RUN rm -rf ~/.cache
RUN rm -rf /root/.cache
RUN mkdir /cache_dir
RUN ln -s /cache_dir ~/.cache
RUN ln -s /cache_dir /root/.cache
ENV XDG_CACHE_HOME=/cache_dir
ENV TORCH_HOME=/cache_dir
ENV MPLCONFIGDIR=/cache_dir

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y sudo
RUN pip3 install --upgrade pip
RUN apt-get install -y ffmpeg

RUN pip3 install -U --no-cache-dir pytorch-fast-transformers
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt


ENV BUILD_MONAI=1
RUN wget https://github.com/Project-MONAI/MONAI/archive/0aa936f87a694a66d54e514ec823a37e999be862.zip && \
    unzip 0aa936f87a694a66d54e514ec823a37e999be862.zip && \
    rm 0aa936f87a694a66d54e514ec823a37e999be862.zip && \
    cd MONAI-0aa936f87a694a66d54e514ec823a37e999be862 && \
    python3 setup.py develop

ENTRYPOINT ["/bin/bash"]
