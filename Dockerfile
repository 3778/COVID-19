FROM ubuntu:18.04

WORKDIR /covid-19

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        python3 \
        python3-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

COPY ./requirements.txt ./

RUN pip3 install --requirement requirements.txt

RUN streamlit version

CMD ["streamlit", "hello"]
