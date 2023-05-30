export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)/src/MAT
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file acc_2gpu.yaml run.py --configs ./checkpoints/MobileFill_Place_seg_2023-05-22-00-27/configs.yaml
