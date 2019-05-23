# Use the NVIDIA tensorflow container as parent.
FROM nvcr.io/nvidia/tensorflow:19.04-py3

# Add meta data.
LABEL description="Container to train and run the DeepParticleNet on systems with a GPU."
LABEL maintainer="Max Frei <max.frei@uni-due.de>"

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

# Install NodeJs to as preparation to install JupyterLab extensions.
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs

# Reinstall pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py --force-reinstall

# Reinstall jupyter.
RUN pip3 install --upgrade --force-reinstall --no-cache-dir jupyterlab

# Install JupyterLab Variable inspector.
RUN jupyter labextension install @lckr/jupyterlab_variableinspector

# Install JupyterLab GoToDefinition.
RUN jupyter labextension install @krassowski/jupyterlab_go_to_definition

# Create and set working directory.
RUN mkdir /tf
WORKDIR /tf

# Start Jupyter notebook.
CMD ["jupyter","notebook","--port=8888","--ip=0.0.0.0","--allow-root","--no-browser","."]
