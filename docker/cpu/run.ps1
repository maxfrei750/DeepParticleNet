$basePath = Join-Path $PSScriptRoot "/../.."

# Run container.
docker run -d --rm --name deepparticlenet --network=bridge -p 8888:8888 -p 6006:6006 -e PASSWORD=$env:UserName --rm -v `"$basePath`":/notebooks maxfrei750/deepparticlenet:tf1.10.0-cpu-py3

# Run tensorboard.
docker exec -d deepparticlenet tensorboard --logdir=/notebooks/logs

# Open a terminal.
docker exec -it deepparticlenet bash