CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=8000 \
train.py \
--batch 16 \
./data/celeba-hq