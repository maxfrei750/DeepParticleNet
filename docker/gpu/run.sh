#!/bin/sh

temp=$( realpath "$0"  )
basePath=$( dirname "$temp" )

nvidia-docker run \
	--rm \
	-it \
	--name deepparticlenet \
	--network=host \
	--volume "$basePath":/tf \
	--volume /media/data_fast/datasets/:/tf/datasets/ \
	--volume /media/data_fast/logs/:/tf/logs/ \
	maxfrei750/deepparticlenet:gpu
