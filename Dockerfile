FROM nvidia/cuda:10.0-devel-ubuntu18.04
# FROM nvidia/cuda:10.0-runtime-ubuntu18.04

SHELL ["/bin/bash", "-c"]

# to avoid SSL CA issues for curl and wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates
RUN update-ca-certificates

# install wget to install miniconda and pyspark
# DEBIAN_FRONTEND : to avoid timezone setting prompt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-tk \
    cmake \
    curl \
    wget \
    firefox \
    apt-utils

# install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

# store all data to /data which is volume mounted from the host local directory 
ENV DATA /data
# store all code to /work which is volume mounted from the host local directory 
ENV WORK /work

# miniconda to install javasdk in requirements-conda.txt
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ${WORK}/miniconda && \
    rm miniconda.sh

ENV PATH ${WORK}/miniconda/bin:$PATH
ENV CONDA_PATH ${WORK}/miniconda

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Set channels
RUN conda config --add channels conda-forge # onnx
RUN conda config --add channels pytorch # pytorch

# install dependencies by conda
COPY requirements_conda.txt requirements_conda.txt
RUN conda install --file requirements_conda.txt

# set JAVA_HOME for Spark
ENV JAVA_HOME ${WORK}/miniconda

# install Spark
RUN wget http://www-eu.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz && \
    tar -xzf spark-2.4.0-bin-hadoop2.7.tgz && \
    mv spark-2.4.0-bin-hadoop2.7 /usr/local/spark
RUN rm spark-2.4.0-bin-hadoop2.7.tgz

RUN pip install onnx
RUN python -m pip install jupyter

# copy ntdemo server code and bokeh_examples
COPY server ${WORK}/server

# copy mecab code and dictionary
COPY asset/chatbot ${WORK}/chatbot
WORKDIR ${WORK}/chatbot/download
RUN tar zxfv mecab-0.996.tar.gz
WORKDIR ${WORK}/chatbot/download/mecab-0.996
RUN ./configure && \
    make && \
    make check
RUN su
RUN make install
WORKDIR ${WORK}/chatbot/download
RUN tar zxfv mecab-ipadic-2.7.0-20070801.tar.gz
WORKDIR ${WORK}/chatbot/download/mecab-ipadic-2.7.0-20070801
RUN ./configure --with-charset=utf-8 && \
    mecab-config --libs-only-L | tee /etc/ld.so.conf.d/mecab.conf && \
    ldconfig && \
    make && \
    make install
RUN exit
WORKDIR ${WORK}/chatbot/download
RUN rm mecab-0.996.tar.gz && \
    rm mecab-ipadic-2.7.0-20070801.tar.gz


ENV NTDEMO_HOST 0.0.0.0
ENV NTDEMO_PORT 3030

EXPOSE ${NTDEMO_PORT}
# 8888: jupyter notebook server
# 8050: dash server
# 5006: bokeh server
EXPOSE 8888
EXPOSE 8050
EXPOSE 5006

WORKDIR ${WORK}/server
CMD ["python", "matrixfactorization_drug_discovery_server.py"]
