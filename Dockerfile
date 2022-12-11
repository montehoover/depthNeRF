FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

WORKDIR /nerf

RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN apt-get -y update
RUN apt-get -y install environment-modules curl wget tree silversearcher-ag git
RUN apt-get -y install ffmpeg libsm6 libxext6

# NeRF Requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Aris Requirements
RUN apt-get -y install libopenexr-dev
RUN pip install OpenEXR
RUN pip install hydra-core pillow opencv-python open3d tqdm plotly
RUN pip uninstall -y ipywidgets
RUN pip install ipywidgets==7.7.2

# copy over repo
# COPY . .
