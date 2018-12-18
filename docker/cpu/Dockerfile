# Use an official tensorflow container as parent.
FROM tensorflow/tensorflow:1.10.0-py3

# Add meta data.
LABEL description="Container to train and run the DeepParticleNet on systems without a GPU."
LABEL maintainer="Max Frei <max.frei@uni-due.de>"

# Install additional dependencies
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

# Set the dark theme for jupyter lab.
COPY themes.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
