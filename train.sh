if [ "$1" == "--machine" ] && [ "$2" == "v3-1" ]; then
    accelerate launch scripts/train.py \
    --run_name decoder_diffusion_v3 \
    --wandb_mode online \
    --train_type decoder \
    --model_type decoder \
    --batch_size 8 \
    --valid_batch_size 8 \
    --lr 4.0e-5 \
    --n_steps 500000
elif [ "$1" == "--machine" ] && [ "$2" == "v3-2" ]; then
    accelerate launch scripts/train.py \
    --run_name encoder_diffusion_v3 \
    --wandb_mode online \
    --train_type encoder \
    --model_type encoder \
    --batch_size 16 \
    --valid_batch_size 16 \
    --lr 8.0e-5 \
    --n_steps 250000
elif [ "$1" == "--machine" ] && [ "$2" == "v3-3" ]; then
    accelerate launch scripts/train_byt5.py \
    --run_name byt5_speaker_v3 \
    --wandb_mode online \
    --add_speaker_embedding
elif [ "$1" == "--machine" ] && [ "$2" == "v3-4" ]; then
    accelerate launch scripts/train_byt5.py \
    --run_name byt5_v3 \
    --wandb_mode online
fi