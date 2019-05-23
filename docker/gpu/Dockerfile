# Use the NVIDIA tensorflow container as parent.
FROM nvcr.io/nvidia/tensorflow:19.04-py3

# Add meta data.
LABEL description="Container to train and run the DeepParticleNet on systems with a GPU."
LABEL maintainer="Max Frei <max.frei@uni-due.de>"

## Install Latex for greek symbols, etc. in graphs.
#RUN apt-get update && apt-get install -yq --no-install-recommends \
#	texlive-fonts-extra \
#	texlive-fonts-recommended \
#	texlive-generic-recommended \
#	texlive-latex-base \
#	texlive-latex-extra \
#	texlive-xetex \
#	dvipng

## Reinstall pip. Else, the next block will throw an error.
#RUN apt-get install -yq --no-install-recommends \
#	curl
#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#RUN python3 get-pip.py --force-reinstall

# Install additional dependencies.
RUN pip --no-cache-dir install \
	cython \
	dill \
	imgaug \
	keras==2.1.3 \
	opencv-python==3.3.0.9 \
	scikit-image \
	pandas \
	seaborn \
	ipywidgets


## Install NodeJs to as preparation to install JupyterLab extensions.
#RUN apt-get update && apt-get install -y --no-install-recommends \
#	python-software-properties
#RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
#RUN apt-get install -y nodejs

## Install JupyterLab Variable inspector.
#RUN jupyter labextension install @lckr/jupyterlab_variableinspector

## Install JupyterLab GoToDefinition.
#RUN jupyter labextension install @krassowski/jupyterlab_go_to_definition

## Install additional tools.
#RUN apt-get update && apt-get install -yq --no-install-recommends \
#	psmisc

# # Install Windows fonts.
# RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections 
# RUN apt-get install -y \
#	ttf-mscorefonts-installer

#EXPOSE 8888
RUN mkdir /tf
WORKDIR /tf
CMD ["jupyter","notebook","--port=8888","--ip=0.0.0.0","--allow-root","--no-browser","."]
