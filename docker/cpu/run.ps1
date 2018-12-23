$basePath = Join-Path $PSScriptRoot "/../.."

# Run container.
docker run `
    --rm `
    --detach `
    --name deepparticlenet `
    --network=bridge `
    --publish 8888:8888 `
    --publish 6006:6006 `
    --env PASSWORD=$env:UserName `
    --volume `"$basePath`":/notebooks `
    maxfrei750/deepparticlenet:tf1.10.0-cpu-py3

# Run tensorboard.
docker exec `
    --detach `
    deepparticlenet `
    tensorboard --logdir=/notebooks/logs

# Open a terminal.
docker exec `
    --interactive `
    --tty `
    deepparticlenet `
    bash   
    