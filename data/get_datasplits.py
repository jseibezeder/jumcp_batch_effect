import pyarrow.parquet as pq
import pandas as pd
from numpy import random
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
import tifffile as tiff
from multiprocessing import Pool
from tqdm import tqdm

SEED = 1234
random.seed(SEED)

def create_datasplits(data_file, split_type="", seed_id=1234, add_val=True):
    #function used to create the datasplit file for training validation and testing
    #the data is split into roughly 70% training, 10% validation and 20% testing
    #there are two types of spliting:
    #   "stratified": this ensures that data from each site and batch are present in trainand test sets
    #   "seperated": with this a whole batch(site) is not present in both training and testing 


    table = pq.read_table(data_file).to_pandas()
    table = table.sample(frac=1, random_state=seed_id).reset_index(drop=True)
    
    classes = table["Metadata_JCP2022"].unique()
    num_batches = len(table["Metadata_Batch"].unique())
    
    if split_type == "stratified":
        #TODO: lets see if we want to implement this
        raise NotImplementedError()

    elif split_type == "seperated":
        unique_batches = table["Metadata_Batch"].unique()
        train_batches, test_batches = train_test_split(unique_batches, test_size=0.2, random_state=seed_id)
        
        temp_table = table[table["Metadata_Batch"].isin(train_batches)]
        test_table = table[table["Metadata_Batch"].isin(test_batches)]
        if add_val:
            train_table, val_table = train_test_split(temp_table, test_size=1/8, random_state=seed_id)
        else:
            train_table = temp_table


    elif split_type == "random":
        #split randomly
        temp_table, test_table = train_test_split(table, test_size=0.2, random_state=seed_id)
        if add_val:
            train_table, val_table = train_test_split(temp_table, test_size=2/3, random_state=seed_id)
        else:
            train_table =temp_table

    train_table.to_csv(f"data/{split_type}_seed{seed_id}_train.csv", index=False)
    if add_val:
        val_table.to_csv(f"data/{split_type}_seed{seed_id}_val.csv", index=False)
    test_table.to_csv(f"data/{split_type}_seed{seed_id}_test.csv", index=False)

    #save mapping of datalabels
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    id_to_class = {idx: cls for idx, cls in enumerate(classes)}

    with open(f"data/class_mapping_seed{seed_id}.json", "w") as f:
        json.dump(class_to_id, f)
    with open(f"data/id_mapping_seed{seed_id}.json", "w") as f:
        json.dump(id_to_class, f)

def load_image(filepath):
    img = tiff.imread(filepath).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))

def get_training_means(train_file, image_path, out_file = "", num_workers=8):
    batchsize = 1024
    table = pd.read_csv(train_file)
    mean = 0
    sum_squard_erros = 0
    n_total = 0

    filepaths = [os.path.join(image_path, f"{sample_id}.jpg") for sample_id in table["Metadata_Sample_ID"]]

    with Pool(num_workers) as pool:
        #welford’s online algorithm
        for batch_start in tqdm(range(0, len(filepaths), batchsize)):
            batch_paths = filepaths[batch_start:min(batch_start + batchsize,len(filepaths))]
            batch_images = np.stack(pool.map(load_image, batch_paths), axis=0)
            B = batch_images.shape[0]

            batch_mean = batch_images.mean(axis=(0, 2, 3))
            batch_var = batch_images.var(axis=(0, 2, 3))

            if n_total == 0:
                mean = batch_mean
                sum_squard_erros = batch_var * B
                n_total = B
            else:
                delta = batch_mean - mean
                mean += delta * B / (n_total + B)
                sum_squard_erros += batch_var * B + delta**2 * n_total * B / (n_total + B)
                n_total += B

    std = np.sqrt(sum_squard_erros / n_total)

    np.savez(out_file,mean=mean, std=std)
    
if __name__ == "__main__":
    filename = "/system/user/publicwork/sanchez/datasets/jumpcp-indices/indices/source_3_filtered_good_batches.pq"
    table = pq.read_table(filename).to_pandas()
    print(len(table))
    
    #create_datasplits(filename, "random", add_val=False)
    #get_training_means("/system/user/studentwork/seibezed/bachelor/data/random_seed1234_train.csv", "/system/user/publicdata/jumpcp/", "data/random_seed1234_norm.npz")
    #data = np.load("/system/user/studentwork/seibezed/bachelor/data/seperated_seed1234_norm.npz")
    #print(tuple(data["mean"]))
    #print(data["mean"])