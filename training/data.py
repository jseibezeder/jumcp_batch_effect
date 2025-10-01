from torch.utils.data import Dataset

class JUMPCPDataset(Dataset):
    def __init__(self, data_file, iamge_path):
        #TODO: need "Metadata_Sample_ID" for image path
        #"Metadata_JCP2022" for applied compound, this is important for classification
        #additionally: "Metadata_Source" for which lab, "Metadata_Batch" for which batch

        pass





def get_jumcp_dataset(args, is_train):
    file = args.train_file if is_train else args.val_file
    folder = args.image_path

    #dataset = 
    pass


    