# Use an official tensorflow container as parent.
FROM tensorflow/tensorflow:1.13.1-py3-jupyter

# Add meta data.
LABEL description="Container to train and run the DeepParticleNet on systems without a GPU."
LABEL maintainer="Max Frei <max.frei@uni-due.de>"

# Install additional dependencies.
RUN pip --no-cache-dir install \
	cython \
	dill \
	imgaug \
	jupyterlab==0.35 \
	keras==2.1.3 \
	opencv-python==3.3.0.9 \
	scikit-image \
	pandas \
	seaborn \
    wget

# Install NodeJs to as preparation to install JupyterLab extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python-software-properties
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs

# Install JupyterLab Variable inspector.
RUN jupyter labextension install @lckr/jupyterlab_variableinspector

# Install JupyterLab GoToDefinition.
RUN jupyter labextension install @krassowski/jupyterlab_go_to_definition

# Install additional tools.
RUN apt-get update && apt-get install -yq --no-install-recommends \
	psmisc
