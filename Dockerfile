FROM continuumio/miniconda3

WORKDIR /app

COPY . .

RUN apt update && \
    apt-get install libgl1-mesa-glx --yes && \
    apt auto-remove
    
RUN conda config --append channels conda-forge && \
    conda config --append channels nvidia && \
    conda config --append channels pytorch && \
    conda config --append channels mamba

RUN conda install --file requirements.txt --yes && \
    conda clean --all --yes

ENTRYPOINT ["bash"]
