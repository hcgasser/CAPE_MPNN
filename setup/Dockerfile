FROM continuumio/miniconda3

ARG REPO
ARG PASSWORD

ENV REPO=${REPO}
ENV PF=/${REPO}
ENV SOFTWARE=${PF}/external
ENV PROGRAMS=/home/benutzer/programs

ENV ALPHAFOLD_DATA=${PF}/data/AF
ENV PYTHONPATH="${PF}/libs"

SHELL ["/bin/bash", "-c"]

RUN echo "root:${PASSWORD}" | chpasswd && \
    apt-get update && \
    apt-get install -y sudo && \
    apt-get install -y gcc g++ && \
    apt-get install -y wget curl unzip git && \
    apt-get install -y make && \
    apt-get install -y vim tmux && \
    apt-get install -y jq && \
    apt-get install -y tcsh

COPY . ${PF}
# RUN git clone https://github.com/hcgasser/${REPO}.git

WORKDIR ${PF}  
    
RUN useradd -m benutzer && \
    echo "benutzer:${PASSWORD}" | chpasswd && \
    usermod -aG sudo benutzer && \
    echo "benutzer ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/benutzer && \
    chown -R benutzer:benutzer ${PF}

USER benutzer

WORKDIR /${REPO}

RUN conda init bash && \
    rm -rf ${PF}/* && \
    mkdir -p ${PROGRAMS}/bin && \
    echo "PATH=${PROGRAMS}/bin:${PATH}" >> /home/benutzer/.bashrc && \
    echo ". ${PF}/tools/set_ENV.sh" >> /home/benutzer/.bashrc && \
    echo "set-option -g default-shell /bin/bash" > /home/benutzer/.tmux.conf

CMD ["/bin/bash"]
