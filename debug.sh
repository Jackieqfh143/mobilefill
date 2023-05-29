export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)/src/MAT
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1
python -m cProfile -s tottime -o profile.stats run.py --configs ./configs/celeba-hq_train_local.yaml