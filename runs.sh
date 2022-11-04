python train.py --load_dataset False --img_width 30 --img_height 30 --data_count 100 --model SimpleCrowdModel --model_save_file test_model.pt --epochs 10 --device cuda:0 --log_wandb False --batch_size 16

python train.py --load_dataset False --img_width 224 --img_height 224 --data_count 1000 --model SimpleCrowdModel --model_save_file test_model.pt --epochs 50 --device cuda:0 --log_wandb True --batch_size 16

python crowd_output.py --img_width 30 --img_height 30 --model SimpleCrowdModel --device cuda:0 --log_wandb False --input_img_path input.jpg --model_load_path test_model.pt