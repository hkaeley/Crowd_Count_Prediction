from argparse import ArgumentParser
from sklearn.metrics import r2_score
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
                self.model = SimpleCrowdModel(in_channels = self.args.in_channels, out_channels = self.args.out_channels, kernel_size = self.args.kernel_size, 
                stride = self.args.stride, padding = self.args.padding,
                initial_height = self.args.img_height, initial_width = self.args.img_width)
            else:
                raise ValueError("Model name not recognized")
        self.model = self.model.to(args.device)
    
    def train(self):
        if self.args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.args.learning_rate)) 
        elif self.args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate)) 
        else:
            raise ValueError("Optimizer arg not recognized")
        
        if self.args.loss_func == "cross_entropy":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif self.args.loss_func == "mse": #want to use mse for regression
            self.loss_function = torch.nn.MSELoss()

        tqdm_bar = tqdm(range(self.args.epochs))
        for epoch_idx in tqdm_bar:
            self.epoch_idx = epoch_idx
            self.model.train() 
            for input, label in zip(self.train_data_x, self.train_data_y): 
                self.optimizer.zero_grad()
                ground_truth = label
                #ground_truth = torch.from_numpy(ground_truth).float()
                img = input
            
                if self.args.model == "SimpleCrowdModel":
                    output = self.model(img)  
                    
                loss = self.loss_function(output, ground_truth) 
                loss.backward()
                self.optimizer.step()
                tqdm_bar.set_description('Epoch: {:d}, loss_train: {:.4f}'.format(self.epoch_idx, loss.detach().cpu().item()))

            if self.epoch_idx % int(self.args.test_step) == 0 or self.epoch_idx == int(self.args.epochs) - 1: #include last epoch as well
                self.evaluate()   

        self.save_model(True) #save model once done training

    def inference(self, x_data, y_data): #use dataloaders here instead once implemented
        y_pred = []
        y_true = []

        for input, label in zip(x_data, y_data): 
            ground_truth = label
            #ground_truth = torch.from_numpy(ground_truth).float()
            img = input
            if self.args.model == "SimpleCrowdModel":
                output = self.model(img)  
                y_true.append(ground_truth)
                y_pred.append(output)
            else:
                raise ValueError('Model not recognized')
            
        return self.metrics(y_true, y_pred, x_data)

    def compute_roc_auc_score(self, y_true, y_pred):
        # if we take any two observations a and b such that a > b, then roc_auc_score is equal to the probability that our model actually ranks a higher than b

        num_same_sign = 0
        num_pairs = 0
        
        for a in range(len(y_true)):
            for b in range(len(y_true)):
                if y_true[a] > y_true[b]: #find pairs of data in which the true value of a is > true value of b
                    num_pairs += 1
                    if y_pred[a] > y_pred[b]: #if predicted value of a is greater then += 1 since we are correct
                        num_same_sign += 1
                    elif y_pred[a] == y_pred[b]: #case in which they are equal
                        num_same_sign += .5
                
        return num_same_sign / num_pairs

    def compute_loss(self, y_true, y_pred):
        agg_loss = 0
        for gt, pred in zip(y_true, y_pred):

            #compute loss
            loss = self.loss_function(pred, gt)
            agg_loss += loss.detach().cpu().item()
        return agg_loss

    def metrics(self, y_true, y_pred, x_data):
        #compute agg loss
        agg_loss = self.compute_loss(y_true, y_pred)

        #compute auc
        auc = self.compute_roc_auc_score(y_true, y_pred)

        #compute r^2 accuracy

        dir_acc = self.compute_dir_acc(y_true, y_pred, x_data)
        
        return {'agg_loss': agg_loss, 'auc': auc, 'r2': r2_score(y_true, y_pred)}


    #runs inference on training and testing sets and collects scores #only log to wanb during eval since thats only when u get a validation loss
    def evaluate(self):
        self.model.eval()
        if self.args.epochs == 0: #if just doing prediction
            train_results = {}
            print('skipping training set.')
        else:
            train_results = self.inference(self.train_data_x, self.train_data_y)
            train_results.update({'train_avg_loss': train_results["agg_loss"]/len(self.train_data_y)})
            train_results.update({'train_auc': train_results["auc"]})
            train_results.update({'train_r2': train_results["r2"]})
            print("train loss: " + str(train_results['train_avg_loss']))
            print("train auc: " + str(train_results['auc']))
            print("train r2: " + str(train_results['r2']))

        val_results = self.inference(self.test_data_x, self.test_data_y)
        val_results.update({'test_avg_loss': val_results["agg_loss"]/len(self.test_data_y)})
        val_results.update({'val_auc': val_results["auc"]})
        val_results.update({'val_r2': val_results["r2"]})
        print("val loss: " + str(val_results['test_avg_loss']))
        print("val auc: " + str(val_results['auc']))
        print("val r2: " + str(val_results['r2']))

        #train_results.update({'epoch': self.epoch_idx})
        val_results.update({'epoch': self.epoch_idx})

        #combine train and val results into one to make logging easier
        #   only log both during inference
        val_results.update(train_results)
        if self.args.log_wandb == "True":
            wandb.log(val_results)
        else:
            print(val_results)

    
    #=================================================================

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
        ap.add_argument('--img_width', type=int, default=60, help="Resized width") #TODO: decide
        ap.add_argument('--img_height', type=int, default=60, help="Resized height") #TODO: decide
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
        ap.add_argument('--device', type=str, default = "cpu")
        ap.add_argument('--test_step', type=int, default = 5)
        ap.add_argument('--batch_size', type=int, default = 16)
        ap.add_argument('--optimizer', type=str, default = "Adam")
        ap.add_argument('--learning_rate', type=float, default = 0.0001)

        ap.add_argument('--log_wandb', type=str, default = "True")
        ap.add_argument('--wandb_project', type=str, default = "test-project")
        ap.add_argument('--wandb_entity', type=str, default = "h199_research")
        
        #include more args here if we decide to do ensemble training?


        args = ap.parse_args()
        trainer = Trainer(args)
        trainer.build_dataset()
        trainer.build_model()
        trainer.train()
        trainer.save_model()