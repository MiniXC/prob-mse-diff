accelerate launch scripts/train.py \
--run_name baseline_decoder \
--wandb_mode online \
--train_type decoder \
--model_type decoder \
--batch_size 8 \
--valid_batch_size 8 \
--lr 4.0e-5 \
--n_steps 500000