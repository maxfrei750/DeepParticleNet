#!/bin/sh

temp=$( realpath "$0"  )
basePath=$( dirname "$temp" )

docker run \
	--runtime=nvidia \
	--rm \
	-it \
	--detach \
	--shm-size=1g \
	--ulimit memlock=-1 \
	--name deepparticlenet \
	--network=host \
	--volume "$basePath":/tf \
	--volume /media/data_fast/datasets/:/tf/datasets/ \
	--volume /media/data_fast/logs/:/tf/logs/ \
	maxfrei750/deepparticlenet:gpu

# Start tensorboard.
docker exec --detach deepparticlenet tensorboard --logdir=/tf/logs

# Reattach to the docker container.
docker attach deepparticlenet
