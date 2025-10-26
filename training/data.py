from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
import tifffile as tiff
import json
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, InterpolationMode, RandomCrop, RandomRotation
from dataclasses import dataclass
import numpy as np
import torch

class JUMPCPDataset(Dataset):
    def __init__(self, data_file, image_path, class_mapping, transforms = None):
        #TODO: need "Metadata_Sample_ID" for image path
        #"Metadata_JCP2022" for applied compound, this is important for classification
        #additionally: "Metadata_Source" for which lab, "Metadata_Batch" for which batch

        table = pd.read_csv(data_file)

        with open(class_mapping, "r") as f:
            class_to_id = json.load(f)

        self.data = table
        self.image_ids = self.data["Metadata_Sample_ID"]
        self.image_path = image_path
        self.class_mapping = class_to_id
        self.number_classes = len(class_to_id)
        self.transforms = transforms

        #variables for grouped sampling
        if "Metadata_Batch" in self.data.columns:
            self.unique_batches = self.data["Metadata_Batch"]
            batch_to_id = {batch: index for index, batch in enumerate(self.unique_batches)}
            self.n_groups = len(batch_to_id)
            self.groups = list(range(self.n_groups))
            self.group_ids = np.array([batch_to_id[batch] for batch in self.data["Metadata_Batch"]])
            self.group_counts, _ = np.histogram(self.group_ids,
                                                bins=range(self.n_groups + 1),
                                                density=False)
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.read_image(self.image_ids[index])

        metadata = {}
        group = None
        if "Metadata_JCP2022" in self.data.columns:
            mol_class = self.class_mapping[self.data["Metadata_JCP2022"][index]]
            

            if "Metadata_Batch" in self.data.columns:
                metadata["batch"] = self.data["Metadata_Batch"][index]
                group = self.group_ids[index]
            if "Metadata_Source" in self.data.columns:
                metadata["source"] = self.data["Metadata_Source"][index] 

            if metadata == {}:
                return image, mol_class
            return image, mol_class, group, metadata
        else:
            return image
        
    @property
    def num_classes(self):
        return self.number_classes


    
    def read_image(self, path):
        filepath = os.path.join(self.image_path, f"{path}.jpg")

        img = tiff.imread(filepath)
        img = np.transpose(img, (2,0,1))
        img = torch.tensor(img, dtype=torch.float32) / 255.0

        if self.transforms:
            image = self.transforms(image)

        return img


#TODO: can add transform functions


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def get_jumcp_data(args, is_train):
    if args.train_indices:
        file = args.train_file
        
    else:
        file = args.train_file if is_train else args.val_file
    folder = args.image_path

    #create transform function
    if args.normalize == "dataset":
        normalization_coefficients = np.load(args.norm_coef_path)
        transforms = transform_function(n_px_tr=args.image_resolution_train,
                                        n_px_val= args.val_image_resolution,
                                        is_train= is_train,
                                        normalize= args.normalize, 
                                        norm_coef = normalization_coefficients,
                                        preprocess=args.preprocess)
    else:
        transforms = transform_function(n_px_tr=args.image_resolution_train,
                                       n_px_val= args.val_image_resolution,
                                       is_train= is_train,
                                       normalize= args.normalize, 
                                       preprocess=args.preprocess)

    if transforms:
        dataset = JUMPCPDataset(file, folder, args.mapping, transforms)
    else:
        dataset = JUMPCPDataset(file, folder, args.mapping)

    if args.train_indices and is_train:
        dataset = Subset(dataset, args.train_indices)
    elif args.val_indices and not is_train:
        dataset = Subset(dataset, args.val_indices)

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset, seed=args.seed) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size if is_train else args.batch_size_eval

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,                            #use for multiprocessing using cuda
        sampler=sampler,
        drop_last=is_train
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_data(args):
    data = {}

    if args.train_file:
        data["train"] = get_jumcp_data(args, is_train=True)
    if args.val_file:
        data["val"] = get_jumcp_data(args, is_train=False)

    return data

def transform_function(n_px_tr: int, n_px_val: int, is_train: bool, normalize:str = "dataset", norm_coef: np.lib.npyio.NpzFile = None, preprocess:str = "downsize"):
    if normalize == "dataset":
        mean = norm_coef["mean"]
        std = norm_coef["std"]
        normalize = Normalize(tuple(mean),tuple(std))
    else:
        normalize = None

    if is_train:
        if preprocess == "crop":
            resize =  RandomCrop(n_px_tr)
        elif preprocess == "downsize":
            resize = RandomResizedCrop(n_px_tr, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC)
        elif preprocess == "rotate":
            resize = Compose([
                              RandomRotation((0, 360)),
                              CenterCrop(n_px_tr)
                            ])
        else: 
            resize = None

    else:
        if preprocess == "crop" or "rotate":
            resize = Compose([
                              CenterCrop(n_px_val),
                              ])
        elif preprocess == "downsize":
            resize = Compose([
                              Resize(n_px_val, interpolation=InterpolationMode.BICUBIC),
                              CenterCrop(n_px_val),
                              ])
        else: 
            resize = None

    if normalize and resize:
        return Compose([
            ToTensor(),
            resize,
            normalize,
        ])
    elif resize:
        return Compose([
            ToTensor(),
            resize,
        ])
    
    return None




if __name__=="__main__":
    filename = "/system/user/studentwork/seibezed/bachelor/data/random_seed1234_train.csv"
    a = JUMPCPDataset(filename,"/system/user/publicdata/jumpcp/", "/system/user/studentwork/seibezed/bachelor/data/class_mapping.json")

    print(a[0])
    