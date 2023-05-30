export PYTHONPATH=$(pwd)
python prepare_masks.py \
--dataset_name "Celeba-hq" \
--mask_type "thick_512" \
--target_size 512 \
--aspect_ratio_kept \
--fixed_size \
--total_num  10000 \
--img_dir "/home/codeoops/CV/data/Celeba-hq/test_512" \
--save_dir "./dataset/validation"