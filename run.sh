export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)/src/MAT
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file acc_1gpu.yaml run.py --configs ./configs/celeba-hq_train_256.yaml
