export PYTHONPATH=$(pwd)
python performance.py \
--dataset_name Place \
--model_path ./checkpoints/place_best.pth \
--mask_type thick_512 \
--target_size 512 \
--total_num 1000 \
--sample_num 3 \
--img_dir "/home/codeoops/CV/InPainting/Inpainting_baseline/compare/results/place_512/(thick_512_10.0k)/real_imgs" \
--mask_dir "/home/codeoops/CV/InPainting/Inpainting_baseline/compare/results/place_512/(thick_512_10.0k)/masks" \
--save_dir ./results \
--aspect_ratio_kept \
--fixed_size \
--center_crop \
--batch_size 10