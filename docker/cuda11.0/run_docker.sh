docker run -e DISPLAY=$DISPLAY \
           --name="qin_simpleclick_dev" \
           -v /tmp/Xauthority-qinliu19:/root/.Xauthority:rw \ # Change to your Xautority file
           --shm-size 16G \
           --gpus all \
           -v /playpen-raid2/qinliu/data:/work/data \ # Mount your data folder
           -v /playpen-raid2/qinliu/projects:/work/projects \  # Mount your code folder
           --net host --rm -it simpleclick:cu110_v1.0 bash
