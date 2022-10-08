import json
from tqdm import tqdm
import sklearn.model_selection
import numpy as np
import sys, os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import pickle as pkl
import json
import cv2

                    
class Dataset():

    def __init__(self, args):
        self.args = args
        self.dataset_path = args.dataset_file_path
        self.dataset_save_path = args.dataset_save_path
        self.width = args.img_width
        self.height = args.img_height
        
        
    #store data into two lists
    def extract_data(self):
        self.images = []
        self.labels = []

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(self.dataset_path)

        image_folder_names = [x for x in Path(self.dataset_path).iterdir() if x.is_dir() and "images" in x.stem]
        json_folder = [x for x in Path(self.dataset_path).iterdir() if x.is_dir() and "jsons" in x.stem][0]
        
        data_count = 0

        for image_folder in image_folder_names:
            images = [Path(f) for f in image_folder.iterdir() if isfile(f) and "jpg" in f.suffix]
            for image_ in tqdm(range(len(images))):
                image = images[image_]
                im = self._load_image(image)
                self.images.append(im)
                image_name = image.stem.split(".")[0]
                self.labels.append(self.find_label(json_folder, image_name))
                data_count += 1
                if self.args.data_count != 0 and data_count == self.args.data_count: #0 will indicate extract all data
                    break
            if self.args.data_count != 0 and data_count == self.args.data_count:
                break

    #find label associated with given image
    def find_label(self, label_folder, image_name):
        label_file_path = (label_folder/(image_name + '.json')).resolve()
        with open(label_file_path) as f:
            data = json.load(f)
            return int(data["human_num"])

    #Represent each image as a tensor            
    def _load_image(self, image_path):
        im = cv2.imread(str(image_path), 1)
        im = cv2.resize(im, (self.width, self.height)) #TODO: see what sizes papers using this dataset use
        self.im_height, self.im_width, self.color_channels = im.shape
        return im

    def split_dataset(self):
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y = sklearn.model_selection.train_test_split(self.images, self.labels, train_size = 0.7, random_state = 42) #split into train and test
        self.images, self.labels = [], []

    #load/save data from dataset_path into data, labels, meta
    def save(self):
        with open(self.dataset_save_path, 'wb') as f:
            pkl.dump(self, f)

    def load(self):
        with open(self.dataset_save_path, 'rb') as f:
            return pkl.load(f)
        
