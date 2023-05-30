export TORCH_HOME=$(pwd)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file acc_2gpu.yaml train_gan.py --configs ./configs/celeba-hq_train_256.yaml
