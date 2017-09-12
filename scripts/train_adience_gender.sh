
python train/model_training.py \
    --weights_path model/age.h5 \
    --lr 0.001 \
    --model adience \
    --data aligned/age.p \
    --epochs 75 \
    --batch_size 64 \
    --num_classes 8
