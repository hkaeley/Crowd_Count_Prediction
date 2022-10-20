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
from model import SimpleCrowdModel

class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        self.data = Dataset(args)
        self.epoch_idx = self.args.epochs
        if self.args.log_wandb == "True":
            wandb.init(project=self.args.wandb_project, entity=self.args.wandb_entity)

    def build_dataset(self):
        if not self.args.load_dataset:
            self.data.extract_data()
            self.data.split_dataset()
            self.data.save()
        else:
            self.data = self.data.load()

    def build_model(self):
        #load model if true
        #else build model depending on args
        #self, in_channels, out_channels, kernel_size, stride, padding)
        if self.args.load_model == "True":
            self.load_model()
        else:
            if self.args.model == "SimpleCrowdModel":
                self.model = SimpleCrowdModel(in_channels = self.args.in_channels, out_channels = self.args.out_channels, kernel_size = self.args.kernel_size, stride = self.args.stride, padding = self.args.padding)
            else:
                raise ValueError("Model name not recognized")
        self.model = self.model.to(args.device)

    def train(self):
        #TODO wandb logging
        pass

    def metrics(self, y_true, y_pred, x_data):
        #TODO
        #R2
        #mae
        pass

    '''run model prediction on the input data'''
    def inference(self, x_data, y_data):
        #TODO
        pass

    '''runs inference on training and testing sets and collects scores for data logging purposes''' #only log to wanb during eval since thats only when u get a validation loss
    def evaluate(self):
        #TODO
        pass

    def save_model(self, is_best=False):
        if is_best:
            saved_path = Path(self.args.model_save_directory + self.args.model_save_file).resolve()
        else:
            saved_path = Path(self.args.model_save_directory + self.args.model_save_file).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save({
            'epoch': self.epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'loss': self.metrics.best_val_loss,
        }, str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.args.__dict__))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))
        print("Model saved.")


    '''Function to load the model, optimizer, scheduler.'''
    def load_model(self):  
        saved_path = Path(self.args.model_load_path).resolve()
        if saved_path.exists():
            self.build_model()
            torch.cuda.empty_cache()
            checkpoint = torch.load(str(saved_path), map_location="cpu")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch_idx = checkpoint['epoch']
            #self.metrics.best_val_loss = checkpoint['loss']
            self.model.to(self.args.device)
            self.model.eval()
        else:
            raise FileNotFoundError("model load path does not exist.")
        print("Model loaded from file.")


if __name__ == "__main__":
        ap = ArgumentParser(description='The parameters for training.')
        ap.add_argument('--load_dataset', type=bool, default = True)
        ap.add_argument('--dataset_file_path', type=str, default="../../dataset", help="The path defining location of indicator dataset.")
        ap.add_argument('--dataset_save_path', type=str, default="../../dataset.pkl", help="The path defining save/load location of processed dataset pkl.")
        ap.add_argument('--img_width', type=int, default=60, help="Resized width")
        ap.add_argument('--img_height', type=int, default=60, help="Resized height")
        ap.add_argument('--data_count', type=int, default=10, help="Number of images to extract")


        ap.add_argument('--model', type=str, default = "SimpleCrowdModel")

        ap.add_argument('--in_channels', type=int, default = 3) #rgb
        ap.add_argument('--out_channels', type=int, default = 20) #play with this
        ap.add_argument('--kernel_size', type=int, default = 3) #play with this
        ap.add_argument('--stride', type=int, default = 1)
        ap.add_argument('--padding', type=int, default = 0) 

        ap.add_argument('--model_save_directory', type=str, default = "saved_models/")
        ap.add_argument('--model_save_file', type=str, default = "best_model.pt")
        ap.add_argument('--model_load_path', type=str, default = "saved_models/best_model.pt")
        ap.add_argument('--load_model', type=bool, default = False)


        ap.add_argument('--epochs', type=int, default = 50)
        ap.add_argument('--device', type=str, default = "cuda:0")
        ap.add_argument('--test_step', type=int, default = 5)
        ap.add_argument('--batch_size', type=int, default = 16)
        ap.add_argument('--optimizer', type=str, default = "Adam")
        ap.add_argument('--learning_rate', type=float, default = 0.0001)

        ap.add_argument('--log_wandb', type=str, default = "True")
        ap.add_argument('--wandb_project', type=str, default = "") #TODO
        ap.add_argument('--wandb_entity', type=str, default = "") #TODO
        
        #include more args here if we decide to do ensemble training?


        args = ap.parse_args()
        trainer = Trainer(args)
        trainer.build_dataset()
        trainer.build_model()
        trainer.train()
        trainer.save_model()