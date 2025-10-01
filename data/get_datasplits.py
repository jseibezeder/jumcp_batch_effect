import pyarrow.parquet as pq
import pandas as pd
from numpy import random
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 1234
random.seed(SEED)

def create_datasplits(data_file, split_type=""):
    #function used to create the datasplit file for training validation and testing
    #the data is split into roughly 70% training, 10% validation and 20% testing
    #there are two types of spliting:
    #   "stratified": this ensures that data from each site and batch are present in trainand test sets
    #   "seperated": with this a whole batch(site) is not present in both training and testing 


    table = pq.read_table(filename).to_pandas()
    table = table.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    num_classes = len(table["Metadata_JCP2022"].unique())
    num_batches = len(table["Metadata_Batch"].unique())
    
    if split_type == "stratified":
        #TODO: lets see if we want to implement this
        pass

    elif split_type == "seperated":
        unique_batches = table["Metadata_Batch"].unique()
        train_batches, test_batches = train_test_split(unique_batches, test_size=0.2, random_state=SEED)
        
        temp_table = table[table["Metadata_Batch"].isin(train_batches)]
        test_table = table[table["Metadata_Batch"].isin(test_batches)]
        train_table, val_table = train_test_split(temp_table, test_size=1/8, random_state=SEED)


    elif split_type == "random":
        #split randomly
        train_table, temp_table = train_test_split(table, test_size=0.3, random_state=SEED)
        val_table, test_table = train_test_split(temp_table, test_size=2/3, random_state=SEED)

    train_table.to_csv(f"data/{split_type}_seed{SEED}_train.csv", index=False)
    val_table.to_csv(f"data/{split_type}_seed{SEED}_val.csv", index=False)
    test_table.to_csv(f"data/{split_type}_seed{SEED}_test.csv", index=False)

if __name__ == "__main__":
    filename = "/system/user/publicwork/sanchez/datasets/jumpcp-indices/indices/source_3_filtered_good_batches.pq"
    create_datasplits(filename, "random")