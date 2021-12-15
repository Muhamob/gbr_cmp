dataset=$1

python /home/isabella/code/competitions/great_barrier_reef/project/reef/yolo/train.py \
    --img 640 \
    --batch 8 \
    --epochs 10 \
    --data /home/isabella/code/competitions/great_barrier_reef/data/splits/$dataset/data.yaml \
    --weights yolov5s.pt \
    --workers 4 \
    --freeze 10 \
    --adam \
    --hyp hyp.yaml
