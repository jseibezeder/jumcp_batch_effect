from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import tifffile as tiff
import json
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
import numpy as np
import torch

class JUMPCPDataset(Dataset):
    def __init__(self, data_file, image_path, class_mapping):
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
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.read_image(self.image_ids[index])
        metadata = {}
        
        if "Metadata_JCP2022" in self.data.columns:
            mol_class = self.class_mapping[self.data["Metadata_JCP2022"][index]]

            if "Metadata_Batch" in self.data.columns:
                metadata["batch"] = self.data["Metadata_Batch"][index]
            if "Metadata_Source" in self.data.columns:
                metadata["source"] = self.data["Metadata_Source"][index] 

            if metadata == {}:
                return image, mol_class
            return image, mol_class, metadata
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

        return img


#TODO: can add transform functions


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def get_jumcp_data(args, is_train):
    file = args.train_file if is_train else args.val_file
    folder = args.image_path
    
    dataset = JUMPCPDataset(file, folder, args.mapping)
    
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


if __name__=="__main__":
    filename = "/system/user/studentwork/seibezed/bachelor/data/random_seed1234_train.csv"
    a = JUMPCPDataset(filename,"/system/user/publicdata/jumpcp/", "/system/user/studentwork/seibezed/bachelor/data/class_mapping.json")

    print(a[0])
    