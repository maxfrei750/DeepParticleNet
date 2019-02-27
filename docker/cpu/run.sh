#!/bin/sh

temp=$( realpath "$0"  )
basePath=$( dirname "$temp" )

docker run \
	--rm \
	-it \
	--name deepparticlenet \
	--network=host \
	--volume "$basePath":/tf/notebooks \
	--volume /media/data_fast/datasets/:/tf/notebooks/datasets/ \
	--volume /media/data_fast/logs/:/tf/notebooks/logs/ \
	maxfrei750/deepparticlenet:gpu
