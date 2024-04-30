import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
import torch.nn as nn
import zipfile
import os
import random
import argparse

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from torchvision.transforms.functional import crop
from torch.utils.data import ConcatDataset
from utils.utils import show_batch


########## CONTINUE MOVING CODE HERE ######### - check howto org this so i can get dataloaders.....google for some file formats...and architectrures...

######### TO DO ########
## checl util functions for aupr/auprin/auprout test - under utils
### use jupyter to test functions
### define train.py and use that to call functions 
### adding in argparser

class IDDataset(Dataset):

    """
    Contains ID dataset for train and validate
    """

    ######################## edit from here #############
    def __init__(self, annotations_file, img_dir, transform=None, bbox = False, target_transform=None):
        super().__init__()
        self.img_labels = annotations_file # modified since loading metadata to load dataset
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = self.img_labels['class_id'].nunique()
        self.bbox = False

        ## code in bounding boxes for use later on ########

        if bbox == True: # generate series
            self.bbox = True
            self.bbox_y = annotations_file['bounding_y'].values
            self.bbox_x = annotations_file['bounding_x'].values
            self.bbox_height = annotations_file['bounding_height'].values
            self.bbox_width = annotations_file['bounding_width'].values
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]) # image name stored in 1st row
        image = read_image(img_path)

        # cropping
        if self.bbox:
            image = crop(image, int(self.bbox_y[idx]), int(self.bbox_x[idx]), int(self.bbox_height[idx]), int(self.bbox_width[idx]))

        # check for greyscale , convert to rgb if so
        if image.shape[0] == 1:
            image = read_image(img_path, mode= ImageReadMode.RGB)

        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

def get_annotations_file(dropped_classes = 40, chosen_split = 0):
    ### generate annotations file for dataset

    """
    returns df['chosen split']
    id and ood df can be obtained by running
    id_df = df[chosen_split]['id']
    ood_df = df[chosen_split]['ood']
    id_df_train, id_df_test, ood_df_train, ood_df_test can be obtained by running
    usually parses directly to 
    """

    ###### preparing metadata ##########
    metadata = pd.read_csv('./meta_data.csv') ######## FIX THIS #####
    # generate sploit of 5 leave out classifiers, each leaving out 40, generae leave out dict
    leave_out = {}
    class_range = np.arange(start= 1, stop = 201)
    ran_state = RandomState(5242) # generate the same splits for some leave out class percentage
    ran_state.shuffle(class_range)
    print(class_range)

    for splits in range(200//dropped_classes):
        leave_out[splits] = class_range[(splits)*dropped_classes:(splits+1)*dropped_classes]

    # generate col for is_oe_image, 0 for ID, 1 for OOD
    df = {}

    for split in leave_out:
        df[split] = {}
        leave_out_condition = metadata["class_id"].isin(leave_out[split])
        leave_in_condition = np.invert(leave_out_condition)
        df[split]['ood'] = metadata.loc[leave_out_condition]
        df[split]['id'] = metadata.loc[leave_in_condition]

    ### renaming class ids for ID data, ood data is not class specific
    # rewrite class from 1 to 160
    for split in df:
        for id_ood in df[split]:
            current_class_arr = df[split][id_ood]['class_id'].unique()
            if id_ood == 'ood':
                future_class_arr = -np.ones(len(current_class_arr))
            else:
                future_class_arr = np.arange(len(current_class_arr)) ###### class labels must span from 0 to 159
            
            replace_dict = {current_class_arr[i]:future_class_arr[i] for i in range(len(current_class_arr))}
            df[split][id_ood]['class_id'].replace(replace_dict, inplace=True)

    return df[chosen_split]

def get_dataset_dict(annotation_file, usage_level, test_size=0.2, data_dir_mode="nscc", custom_data_dir = None):

    """
    annotation file use the prev get function, usage level from arg parse
    rmb custom_data_dir only for is datadir mode = "custom", other wise only "local" or "nscc" allowed

    transforms are fixed
    usage: ranges from 1 to 4
    """

    id_df = annotation_file['id']
    ood_df = annotation_file['ood']

    id_df_train = id_df[id_df['is_training_image'] == 1]
    id_df_test = id_df[id_df['is_training_image'] == 0]
    id_df_test, id_df_val = train_test_split(id_df_test, test_size=test_size, random_state=42)

    ood_df_train = ood_df[ood_df['is_training_image'] == 1]
    ood_df_test = ood_df[ood_df['is_training_image'] == 0]
    
    ### defining transformations

    # defining diff transforms, moved some transforms here
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # taking test_transfrom from sx

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform_more = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(-5, 5, -5, 5)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # defining image directories

    if data_dir_mode == "local":
        img_directory = "/Users/xy/Downloads/data/CUB_200_2011/CUB_200_2011/images"
    elif data_dir_mode == "nscc":
        img_directory = "/home/users/nus/e1144115/scratch/CUB_200_2011/CUB_200_2011/images/"
    elif data_dir_mode == "custom":
        img_directory = custom_data_dir 
    ####### be sure to replace with /home/users/nus/e1144115/scratch/CUB_200_2011/CUB_200_2011/images/
    ## /home/users/nus/e1144115/scratch/CUB_200_2011/CUB_200_2011/images/
    # /content/drive/MyDrive/nus_cs_classes/CS5242/CUB_200_2011/CUB_200_2011/images/
    # "/Users/xy/Downloads/data/CUB_200_2011/CUB_200_2011/images"
    # pls remember to put backslashes

    ### defining dataloaders, 

    ###### creating datasets ######
    train_dataset_aug = IDDataset(id_df_train, img_directory, train_transform, bbox=True) # original with cropping
    train_dataset_augnocrop = IDDataset(id_df_train, img_directory, train_transform) ## no crop
    train_dataset_augmore = IDDataset(id_df_train, img_directory, train_transform_more, bbox=True) ### added aug
    train_dataset_noaug = IDDataset(id_df_train, img_directory, transform) ## no crop

    test_dataset_aug = IDDataset(id_df_test, img_directory, test_transform, bbox=True)
    test_dataset_augnocrop = IDDataset(id_df_test, img_directory, test_transform)
    test_dataset_augmore = IDDataset(id_df_test, img_directory, test_transform, bbox=True)
    test_dataset_noaug = IDDataset(id_df_test, img_directory, test_transform)

    val_dataset_aug = IDDataset(id_df_val, img_directory, test_transform, bbox=True)
    val_dataset_augnocrop = IDDataset(id_df_val, img_directory, test_transform)
    val_dataset_augmore = IDDataset(id_df_val, img_directory, test_transform, bbox=True)
    val_dataset_noaug = IDDataset(id_df_val, img_directory, test_transform)

    oe_dataset_aug = IDDataset(ood_df_train, img_directory, train_transform, bbox=True) # original with cropping
    oe_dataset_augnocrop = IDDataset(ood_df_train, img_directory, train_transform) ## no crop
    oe_dataset_augmore = IDDataset(ood_df_train, img_directory, train_transform_more, bbox=True) ### added aug
    oe_dataset_noaug = IDDataset(ood_df_train, img_directory, transform) ## no crop

    oeval_dataset_aug = IDDataset(ood_df_test, img_directory, test_transform, bbox=True)
    oeval_dataset_augnocrop = IDDataset(ood_df_test, img_directory, test_transform)
    oeval_dataset_augmore = IDDataset(ood_df_test, img_directory, test_transform, bbox=True)
    oeval_dataset_noaug = IDDataset(ood_df_test, img_directory, test_transform)

    train_dataset = ConcatDataset([train_dataset_aug, train_dataset_augnocrop, train_dataset_augmore, train_dataset_noaug][:usage_level])
    test_dataset = ConcatDataset([test_dataset_aug, test_dataset_augnocrop, test_dataset_augmore, test_dataset_noaug][:usage_level])
    val_dataset = ConcatDataset([val_dataset_aug, val_dataset_augnocrop, val_dataset_augmore, val_dataset_noaug][:usage_level])
    oe_dataset = ConcatDataset([oe_dataset_aug, oe_dataset_augnocrop, oe_dataset_augmore, oe_dataset_noaug][:usage_level])
    oeval_dataset = ConcatDataset([oeval_dataset_aug, oeval_dataset_augnocrop, oeval_dataset_augmore, oeval_dataset_noaug][:usage_level])
    final_val_dataset = ConcatDataset([oeval_dataset, val_dataset, test_dataset])
    print(f"len of train_Dataset: {len(train_dataset)}, len of test_dataset {len(test_dataset)}, len of val_dataset: {len(val_dataset)}, len of oe_dataset:{len(oe_dataset)}, len of oeval_dataset:{len(oeval_dataset)}, len of final val mix:{len(final_val_dataset)}")

    dataset_dict = {"train_dataset": train_dataset, "test_dataset": test_dataset, "val_dataset": val_dataset,
                    "oe_dataset": oe_dataset, "oeval_dataset": oeval_dataset, "final_val_dataset": final_val_dataset}
    
    return dataset_dict


def get_dataloader_dict(dataset_dict):

    """
    dataset dict is smth like     
    dataset_dict = {"train_dataset": train_dataset, "test_dataset": test_dataset, "val_dataset": val_dataset,
                    "oe_dataset": oe_dataset, "oeval_dataset": oeval_dataset, "final_val_dataset": final_val_dataset}
    returns dataloader dict in similar format
    """

    train_loader = DataLoader(dataset_dict["train_dataset"], batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_dict["test_dataset"], batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset_dict["val_dataset"], batch_size=64, shuffle=True)
    oe_loader = DataLoader(dataset_dict["oe_dataset"], batch_size=16, shuffle=True)
    oeval_loader = DataLoader(dataset_dict["oeval_dataset"], batch_size=16, shuffle=True)
    final_val_loader = DataLoader(dataset_dict["final_val_dataset"], batch_size=64, shuffle=True)
    print(f"len of train_loader: {len(train_loader)}, len of test_datalaoder {len(test_loader)}, len of val_datalaoder: {len(val_loader)}, len of oe_datalaoder:{len(oe_loader)}, len of oeval_loader:{len(oeval_loader)}, len of final val mix:{len(final_val_loader)}")

    dataloader_dict = {"train": train_loader, "test": test_loader, "val": val_loader,
                "oe": oe_loader, "oeval": oeval_loader, "final_val": final_val_loader}
    
    return dataloader_dict
