from argparse import ArgumentParser
import torch
from tqdm import tqdm
import numpy as np
import sys, os
from pathlib import Path
import wandb 

sys.path.append("../data")
from data import Dataset
sys.path.append("../model")
from model import *

class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        self.data = Dataset(args)

    def build_dataset(self):
        if not self.args.load_dataset:
            self.data.extract_data()
            self.data.split_dataset()
            self.data.save()
        else:
            self.data = self.data.load()

    def build_model(self):
        #TODO
        pass

    def train(self):
        #TODO
        pass

if __name__ == "__main__":
        ap = ArgumentParser(description='The parameters for training.')
        ap.add_argument('--load_dataset', type=bool, default = True)
        ap.add_argument('--dataset_file_path', type=str, default="../../dataset", help="The path defining location of indicator dataset.")
        ap.add_argument('--dataset_save_path', type=str, default="../../dataset.pkl", help="The path defining save/load location of processed dataset pkl.")
        ap.add_argument('--img_width', type=int, default=60, help="Resized width")
        ap.add_argument('--img_height', type=int, default=60, help="Resized height")
        ap.add_argument('--data_count', type=int, default=10, help="Number of images to extract")


        ap.add_argument('--epochs', type=int, default = 50)
        ap.add_argument('--device', type=str, default = "cuda:0")
        ap.add_argument('--test_step', type=int, default = 5)
        ap.add_argument('--batch_size', type=int, default = 16)
        ap.add_argument('--optimizer', type=str, default = "Adam")
        ap.add_argument('--learning_rate', type=float, default = 0.0001)

        ap.add_argument('--log_wandb', type=str, default = "True")
        ap.add_argument('--wandb_project', type=str, default = "") #TODO
        ap.add_argument('--wandb_entity', type=str, default = "") #TODO

        ap.add_argument('--model', type=str, default = "CrowdModel")
        ap.add_argument('--model_save_directory', type=str, default = "saved_models/")
        ap.add_argument('--model_save_file', type=str, default = "best_model.pt")
        ap.add_argument('--model_load_path', type=str, default = "saved_models/best_model.pt")
        ap.add_argument('--load_model', type=bool, default = False)
        
        #include more args here if we decide to do ensemble training?


        args = ap.parse_args()
        trainer = Trainer(args)
        trainer.build_dataset()
        trainer.build_model()
        trainer.train()


if __name__ == "__main__":
    pass