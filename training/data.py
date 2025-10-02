from torch.utils.data import Dataset
import pandas as pd
import os
import tifffile as tiff


class JUMPCPDataset(Dataset):
    def __init__(self, data_file, image_path, num_classes):
        #TODO: need "Metadata_Sample_ID" for image path
        #"Metadata_JCP2022" for applied compound, this is important for classification
        #additionally: "Metadata_Source" for which lab, "Metadata_Batch" for which batch

        table = pd.read_csv(data_file)

        self.data = table
        self.image_ids = self.data["Metadata_Sample_ID"]
        self.image_path = image_path
        self.num_classes = num_classes
        #TODO: find a mapping from calsses to numbers
   
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, index):
        image = self.read_image(self.image_ids[index])
        if "Metadata_JCP2022" in self.data.columns:
            mol_class = self.data["Metadata_JCP2022"][index]
            #TODO: add mapping

            return image, mol_class
        else:
            return image


    
    def read_image(self, path):
        filepath = os.path.join(self.image_path, f"{path}.jpg")

        img = tiff.imread(filepath)

        return img


#TODO: can add transform functions




def get_jumcp_dataset(args, is_train):
    file = args.train_file if is_train else args.val_file
    folder = args.image_path

    #dataset = 
    pass


if __name__=="__main__":
    filename = "/system/user/studentwork/seibezed/bachelor/data/random_seed1234_train.csv"
    a = JUMPCPDataset(filename,"/system/user/publicdata/jumpcp/", 8)

    print(a[0])
    