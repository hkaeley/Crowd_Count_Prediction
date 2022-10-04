from dataset import Dataset
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from model import StockFormer, lstm, StockFormerEncoder
import numpy as np
import sys, os
from pathlib import Path
import wandb 

sys.path.append("../")
from model import *
from train import *

class Trainer():
    def __init__(self) -> None:
        pass

    def build_dataset():
        pass
    def split_dataset():
        pass
    def build_model():
        pass
    def train():
        pass

if __name__ == "__main__":
        ap = ArgumentParser(description='The parameters for training.')
        ap.add_argument('--dataset_file_path', type=str, default="C:/Users/harsi/research/stockformer/StockFormer/data.csv", help="The path defining location of indicator dataset.")
        ap.add_argument('--epochs', type=int, default = 50)
        ap.add_argument('--device', type=str, default = "cuda:0")
        ap.add_argument('--test_step', type=int, default = 5)
        ap.add_argument('--batch_size', type=int, default = 16)
        ap.add_argument('--optimizer', type=str, default = "Adam")
        ap.add_argument('--learning_rate', type=float, default = 0.0001)
        ap.add_argument('--log_wandb', type=str, default = "True")

        ap.add_argument('--wandb_project', type=str, default = "") #TODO
        ap.add_argument('--wandb_entity', type=str, default = "") #TODO
        ap.add_argument('--model_save_directory', type=str, default = "saved_models/")
        ap.add_argument('--model_save_file', type=str, default = "model_best_val_loss_.vec.pt")
        ap.add_argument('--model_load_path', type=str, default = "saved_models/model_best_val_loss_.vec.pt")
        ap.add_argument('--load_model', type=bool, default = False)
        ap.add_argument('--model', type=str, default = "stockformer")
        #include more args here if we decide to do ensemble training?


        args = ap.parse_args()
        trainer = Trainer(args)
        trainer.build_dataset()
        trainer.split_dataset()
        trainer.build_model()
        trainer.train()


if __name__ == "__main__":
    pass