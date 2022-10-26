docker run -e DISPLAY=$DISPLAY \
           --name="qin_simpleclick_dev" \
           -v /tmp/Xauthority-qinliu19:/root/.Xauthority:rw \
           --shm-size 16G \
           --gpus all \
           -v /playpen-raid2/qinliu/data:/work/data \
           -v /playpen-raid2/qinliu/projects:/work/projects \
           --net host --rm -it simpleclick:cu116_v1.0 bash
