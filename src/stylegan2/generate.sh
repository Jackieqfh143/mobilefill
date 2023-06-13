export PYTHONPATH=$(dirname $(dirname "$PWD"))
python generate.py --sample 1 --pics 10 --ckpt ./checkpoint/stylegan2-church-config-f.pt --size 256