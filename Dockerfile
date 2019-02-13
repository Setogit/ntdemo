FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

# install pip
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# RUN python get-pip.py

# install wget to install miniconda and pyspark
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    wget

# to avoid SSL CA issues
RUN update-ca-certificates

ENV HOME /home
WORKDIR ${HOME}/

# miniconda to install javasdk in requirements-conda.txt
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ${HOME}/miniconda && \
    rm miniconda.sh

ENV PATH ${HOME}/miniconda/bin:$PATH
ENV CONDA_PATH ${HOME}/miniconda

COPY server ${HOME}/server
WORKDIR ${HOME}/server

# Set channels
RUN conda config --add channels conda-forge # onnx
RUN conda config --add channels pytorch # pytorch

# install dependencies by conda
# ADD requirements_conda.txt requirements_conda.txt
RUN conda install --file requirements_conda.txt
RUN rm requirements_conda.txt

RUN pip install onnx

# set JAVA_HOME for Spark
ENV JAVA_HOME ${HOME}/miniconda

# install Spark
RUN wget http://www-eu.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz && \
    tar -xzf spark-2.4.0-bin-hadoop2.7.tgz && \
    mv spark-2.4.0-bin-hadoop2.7 /usr/local/spark
RUN rm spark-2.4.0-bin-hadoop2.7.tgz

RUN pip install -r requirements.txt
RUN rm requirements.txt

ENV NTDEMO_HOST 0.0.0.0
ENV NTDEMO_PORT 3030

EXPOSE 3030

ENTRYPOINT ["python"]
CMD ["matrixfactorization_drug_discovery_server.py"]
