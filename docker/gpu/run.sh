#!/bin/sh

temp=$( realpath "$0"  )
basePath=$( dirname "$temp" )/../..

nvidia-docker run \
	--rm \
	--detach \
	--name deepparticlenet \
	--network=host --env PASSWORD=$USER \
	--volume "$basePath":/notebooks \
	maxfrei750/deepparticlenet:tf1.10.0-gpu-py3

# Run tensorboard.
docker exec \
	--detach \
	deepparticlenet \
	tensorboard --logdir=/notebooks/logs

# Open a terminal.
docker exec \
	--interactive \
	--tty \
	deepparticlenet \
	bash
