# Use an official tensorflow container as parent.
FROM tensorflow/tensorflow:1.10.0-gpu-py3

# Add meta data.
LABEL description="Container to train and run the DeepParticleNet on systems with a GPU."
LABEL maintainer="Max Frei <max.frei@uni-due.de>"

# Install additional dependencies.
RUN pip --no-cache-dir install \
	cython \
	dill \
	imgaug \
	IPython[all] \
	jupyterlab \
	keras==2.1.3 \
	opencv-python==3.3.0.9 \
	scikit-image==0.13

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Prepare to install stuff with apt-get
RUN apt-get update

# Install variable inspector extension for jupyterlab.
RUN apt-get install -y --no-install-recommends \
    curl \
    python-software-properties
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs
RUN jupyter labextension install @lckr/jupyterlab_variableinspector

# Install Latex for greek symbols, etc. in graphs.
RUN apt-get install -y --no-install-recommends \
	texlive-latex-extra \
	dvipng