[![Paper (Powder Technology)](https://img.shields.io/badge/DOI-10.1016/j.powtec.2019.10.020-blue.svg)](https://doi.org/10.1016/j.powtec.2019.10.020)
[![Paper (arXiv)](https://img.shields.io/badge/arXiv-1907.05112-b31b1b.svg)](https://arxiv.org/abs/1907.05112)
[![License](https://img.shields.io/github/license/maxfrei750/DeepParticleNet.svg)](https://github.com/maxfrei750/DeepParticleNet/blob/master/LICENSE) 
[![Docker Cloud Automated build](https://img.shields.io/docker/cloud/automated/maxfrei750/deepparticlenet.svg)](https://hub.docker.com/r/maxfrei750/deepparticlenet)
[![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/maxfrei750/deepparticlenet.svg)](https://hub.docker.com/r/maxfrei750/deepparticlenet)

# DeepParticleNet

This repository is a toolbox for the easy, deep learning-based primary particle size analysis of agglomerated, aggregated, partially sintered or simply occluded particles. It accompanies the following publication:

[Image-Based Size Determination of Agglomerated 
and Partially Sintered Particles via Convolutional Neural Networks](https://doi.org/10.1016/j.powtec.2019.10.020)

The utilized convolutional neural network was inspired by the Mask R-CNN architecture, developed by [He et al.](https://arxiv.org/abs/1703.06870) and is based on an implementation of [Abdulla](https://github.com/matterport/Mask_RCNN), realized with Keras and TensorFlow, controlled by Python.

## Table of Contents
   * [DeepParticleNet](#DeepParticleNet)
   * [Table of Contents](#table-of-contents)
   * [Examples](#examples)
   * [Citation](#citation)
   * [Setup](#setup)
   * [Getting started](#getting-started)

## Examples 
#### Detection
<img src="assets\example_detection.jpg" alt="Example Detection" width="804" height="300"/> 

#### PSD Measurement
<img src="assets\example_psd.png" alt="Example PSD Measurement" width="400" height="265"/>

## Citation
If you use this repository for a publication, then please cite it using the following bibtex-entry:
```
@article{Frei.2019,
    author = {Frei, Max and Kruis, Frank Einar},
    year = {2019},
    title = {Image-Based Size Analysis of Agglomerated and Partially Sintered Particles via Convolutional Neural Networks},
    url = {https://doi.org/10.1016/j.powtec.2019.10.020}
}
```

## Setup

#### CPU only (Linux & Windows)
<details>
<summary>Click to expand ...</summary>

1. Install [docker](https://www.docker.com/).
2. Open a command line.
3. Clone this repository: ``git clone --recurse-submodules https://github.com/maxfrei750/DeepParticleNet.git``
4. Change into the folder of the repository: ``cd DeepParticleNet``
5. Spin up the docker container (adjust paths according to your folder structure):
```
docker run -i --name deepparticlenet -p 8888:8888 -p 6006:6006 -v /path/to/code:/tf -v /path/to/datasets:/tf/datasets -v /path/to/logs:/tf/logs maxfrei750/deepparticlenet:cpu
```

**Optional:** Start Tensorboard
1. Open a command line.
2. Start Tensorboard: ``docker exec -i deepparticlenet tensorboard --logdir=/tf/logs``
3. Access ``localhost:6006`` in your browser.

</details>

#### GPU support (Linux)
<details>
<summary>Click to expand ...</summary>

1. Install [docker](https://www.docker.com/).
2. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
3. Open a command line.
4. Clone this repository: ``git clone --recurse-submodules https://github.com/maxfrei750/DeepParticleNet.git``
5. Change into the folder of the repository: ``cd DeepParticleNet``
6. Spin up the docker container (adjust paths according to your folder structure):
```
nvidia-docker run -i --shm-size=1g --ulimit memlock=-1 --name deepparticlenet -p 8888:8888 -p 6006:6006 -v /path/to/code:/tf -v /path/to/datasets:/tf/datasets -v /path/to/logs:/tf/logs maxfrei750/deepparticlenet:gpu
```

**Optional:** Start Tensorboard
1. Open a command line.
2. Start Tensorboard: ``docker exec -i deepparticlenet tensorboard --logdir=/tf/logs``
3. Access ``localhost:6006`` in your browser.

</details>

#### GPU support (Windows)
<details>
<summary>Click to expand ...</summary>

Nvidia-docker does not support Windows. Therefore, if you are running Windows and need GPU support, then you need to setup a python environment (e.g. conda).

1. Install [conda](https://conda.io/en/latest/miniconda.html).
2. Open a command line.
3. Clone this repository: ``git clone --recurse-submodules https://github.com/maxfrei750/DeepParticleNet.git``
4. Change into the folder of the repository: ``cd DeepParticleNet``
5. Create a new conda environment: 
``conda env create --file dpn-gpu-environment.yml``
6. Activate the new conda environment: ``activate dpn-gpu-env``
7. Start jupyter lab: ``jupyter lab``

**Optional:** Start Tensorboard
1. Open a command line.
3. Activate the conda environment: ``activate dpn-gpu-env``
4. Start Tensorboard: ``tensorboard --logdir=/path/to/logs``
5. Access ``localhost:6006`` in your browser.

</details>

## Getting started
1. Copy the jupyter token from your command line.
2. Enter the jupyter server by accessing ``localhost:8888/lab`` in your browser and pasting the jupyter token that you just copied.
