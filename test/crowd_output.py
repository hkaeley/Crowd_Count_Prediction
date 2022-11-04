from argparse import ArgumentParser
import sys, os
import torch
sys.path.append("../train")
from train import Trainer

if __name__ == "__main__":


    ap = ArgumentParser(description='The parameters for training.')
    ap.add_argument('--img_width', type=int, default=60, help="Resized width") #TODO: decide
    ap.add_argument('--img_height', type=int, default=60, help="Resized height") #TODO: decide
    ap.add_argument('--model', type=str, default = "SimpleCrowdModel")
    ap.add_argument('--in_channels', type=int, default = 3) #rgb
    ap.add_argument('--out_channels', type=int, default = 20) #play with this
    ap.add_argument('--kernel_size', type=int, default = 3) #play with this
    ap.add_argument('--stride', type=int, default = 1)
    ap.add_argument('--padding', type=int, default = 0) 
    ap.add_argument('--device', type=str, default = "cpu")
    ap.add_argument('--test_step', type=int, default = 5)
    ap.add_argument('--batch_size', type=int, default = 8)
    ap.add_argument('--optimizer', type=str, default = "Adam")
    ap.add_argument('--learning_rate', type=float, default = 0.0001)
    ap.add_argument('--model_load_path', type=str, default = "saved_models/best_model.pt")
    ap.add_argument('--log_wandb', type=str, default = "True")
    ap.add_argument('--input_img_path', type=str, default = "input.jpg")


    ap.add_argument('--load_dataset', type=str, default = True)
    ap.add_argument('--dataset_file_path', type=str, default="D:/dataset", help="The path defining location of indicator dataset.")
    ap.add_argument('--dataset_save_path', type=str, default="../../dataset.pkl", help="The path defining save/load location of processed dataset pkl.")
    ap.add_argument('--data_count', type=int, default=10, help="Number of images to extract")
    ap.add_argument('--model_save_directory', type=str, default = "saved_models/")
    ap.add_argument('--model_save_file', type=str, default = "best_model.pt")
    ap.add_argument('--epochs', type=int, default = 50)
    ap.add_argument('--wandb_project', type=str, default = "test-project")
    ap.add_argument('--wandb_entity', type=str, default = "h199_research")
    
    #include more args here if we decide to do ensemble training?


    args = ap.parse_args()
    trainer = Trainer(args)
    trainer.load_model()
    img = trainer.data._load_image(args.input_img_path)
    img = torch.from_numpy(img).float().to(args.device).unsqueeze(0).permute(0, 3, 1, 2)
    print(trainer.model(img).item())