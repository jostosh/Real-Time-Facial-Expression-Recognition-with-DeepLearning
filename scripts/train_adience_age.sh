
python model/model_training.py \
    --weights_path model/gender.h5 \
    --lr 0.001 \
    --model adience \
    --data aligned/gender.p \
    --epochs 75 \
    --batch_size 64 \
    --num_classes 2
