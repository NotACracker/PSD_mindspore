python -B train.py --device_target GPU --device_id 0 --batch_size 3 --labeled_point 1% --scale --name psd_Area-5-gpu --output_dir ./runs
python -B test.py --model_path runs/psd_Area-5-gpu --device_id 0 --device_target gpu
