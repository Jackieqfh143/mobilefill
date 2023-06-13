export TORCH_HOME=$(pwd)
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file acc_1gpu.yaml train_gan.py --configs ./configs/celeba-hq_train_256.yaml
