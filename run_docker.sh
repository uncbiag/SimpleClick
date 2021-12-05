docker run -e DISPLAY=$DISPLAY \
           -v /tmp/Xauthority-qinliu19:/root/.Xauthority:rw \
           --shm-size 8G \
           -v /playpen-raid/qinliu/projects/ritm_interactive_segmentation:/work/ritm \
           -v /playpen-raid/qinliu/data:/work/data \
           --gpus all \
           --net host --rm -it ritm bash
