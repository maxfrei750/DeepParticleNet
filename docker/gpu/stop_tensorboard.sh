# Stop tensorboard.
docker exec --detach deepparticlenet fuser -k 6006/tcp
