export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)/src/MAT
CUDA_VISIBLE_DEVICES=0
python -m debugpy --listen 0.0.0.0:5678 run.py --configs ./configs/celeba-hq_train_local.yaml
