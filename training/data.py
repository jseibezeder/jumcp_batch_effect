from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
import tifffile as tiff
import json
from torch.utils.data.distributed import DistributedSampler
from training.sampler import DistributedGroupSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, InterpolationMode, RandomCrop, RandomRotation
from dataclasses import dataclass
import numpy as np
import torch

from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Dict
)

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
            self.unique_batches = self.data["Metadata_Batch"].unique()
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
            img = self.transforms(img)

        return img


#TODO: can add transform functions


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def get_jumcp_data(args, is_train):
    if args.train_indices is not None:
        file = args.train_file
        
    else:
        file = args.train_file if is_train else args.val_file
    folder = args.image_path

    #create transform function
    if args.normalize == "dataset":
        normalization_coefficients = np.load(args.norm_coef_path)
        transforms = transform_function(n_px_tr=args.image_resolution_train,
                                        n_px_val= args.image_resolution_val,
                                        is_train= is_train,
                                        normalize= args.normalize, 
                                        norm_coef = normalization_coefficients,
                                        preprocess=args.preprocess_img)
    else:
        transforms = transform_function(n_px_tr=args.image_resolution_train,
                                       n_px_val= args.image_resolution_val,
                                       is_train= is_train,
                                       normalize= args.normalize, 
                                       preprocess=args.preprocess_img)

    if transforms:
        dataset = JUMPCPDataset(file, folder, args.mapping, transforms)
    else:
        dataset = JUMPCPDataset(file, folder, args.mapping)

    if args.train_indices is not None and is_train:
        dataset = CustomSubset(dataset, args.train_indices)
    elif args.val_indices is not None and not is_train:
        dataset = CustomSubset(dataset, args.val_indices)

    num_samples = len(dataset)
    batch_size = args.batch_size if is_train else args.batch_size_eval
    
    if args.method == "standard":
        sampler = DistributedSampler(dataset, seed=args.seed) if args.distributed and is_train else None
        shuffle = is_train and sampler is None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,                            #use for multiprocessing using cuda
            sampler=sampler,
            drop_last=is_train
        )
    elif args.method == "armbn" or args.method == "armll":
        sampler = DistributedGroupSampler(dataset, 
                                          meta_batch_size=args.meta_batch_size,
                                          support_size=args.support_size,
                                          seed=args.seed)
        dataloader = DataLoader(
            dataset,
            num_workers=args.workers,
            pin_memory=True,                            #use for multiprocessing using cuda
            batch_sampler=sampler
        )
    elif args.method == "memo":
        raise NotImplementedError()

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_data(args):
    data = {}

    if args.train_file:
        data["train"] = get_jumcp_data(args, is_train=True)
    if args.val_file or args.val_indices is not None:
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
            resize,
            normalize,
        ])
    elif resize:
        return Compose([
            resize,
        ])
    
    return None

T_co = TypeVar('T_co', covariant=True)
class CustomSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.n_groups = dataset.n_groups
        self.groups = dataset.groups
        self.group_ids = dataset.group_ids[indices]
        self.group_counts, _ = np.histogram(self.group_ids,
                                            bins=range(self.n_groups + 1),
                                            density=False)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)


if __name__=="__main__":
    filename = "/system/user/studentwork/seibezed/bachelor/data/random_seed1234_train.csv"
    a = JUMPCPDataset(filename,"/system/user/publicdata/jumpcp/", "/system/user/studentwork/seibezed/bachelor/data/class_mapping.json")

    print(a[0])
    