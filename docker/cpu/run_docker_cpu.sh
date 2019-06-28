#!/bin/sh

temp=$( realpath "$0"  )
basePath=$( dirname "$temp" )

docker run \
	--rm \
	--interactive \
	--detach
	--name deepparticlenet \
	--network=host \
	--volume "$basePath":/tf \
	--volume /media/data_fast/datasets/:/tf/datasets/ \
	--volume /media/data_fast/logs/:/tf/logs/ \
	maxfrei750/deepparticlenet:cpu

# Start tensorboard.
docker exec --detach deepparticlenet tensorboard --logdir=/tf/logs

# Reattach to the docker container.
docker attach deepparticlenet
