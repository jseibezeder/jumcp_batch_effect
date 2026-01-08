import sys
import os
src_path = os.path.split(os.getcwd())[0]
sys.path.insert(0, src_path)

import numpy as np
import torch
import os
import json
from training.model import ResNet, MLP, ContextNet
from torch.utils.data import DataLoader
from argparse import Namespace
from training.data import JUMPCPDataset, create_datasplits, transform_function, CustomSubset
from tqdm import tqdm
import pandas as pd
from training.train import predict_and_loss
from training.sampler import GroupSampler
import torch.nn as nn


SEED =1234

def load_model(args, checkpoint_path, device, model_name, use_running=True):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k[len('module.'):]: v for k,v in state_dict.items()}

    if args.method == "armcml" or args.method == "armll":
        armnet_state_dict = checkpoint["armnet_state_dict"]
        new_armnet_state_dict = {k[len('module.'):]: v for k,v in armnet_state_dict.items()}

    model_config_file = os.path.join(src_path, f"training/model_parameter/{model_name.replace('/', '-')}.json")
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    if args.method == "armcml":
        model_info["input_shape"] += args.n_context_channels
    
    model = ResNet(**model_info, use_batch_running=use_running)
    armnet = None

    if str(device) == "cpu":
        model.float()
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    if args.method=="armll":
        armnet = MLP(in_size=8, norm_reduce=True)
        armnet.load_state_dict(new_armnet_state_dict)
        armnet.to(device)
    elif args.method=="armcml":
        armnet = ContextNet(5, 5, hidden_dim=64, kernel_size=5).to(device)
        armnet.load_state_dict(new_armnet_state_dict)
        armnet.to(device)

    return model, armnet



def main(test_file, model_path, model_name, img_path, mapping_path, args, cv_folds=5, loss_fn = nn.CrossEntropyLoss()):
    data_args = Namespace(
    split_type = "seperated",
    add_val = False,
    train_file = test_file,
    cross_validation = cv_folds
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(torch.cuda.device_count())

    data = []
    transforms = transform_function(n_px_tr=250, n_px_val=250, is_train=False, preprocess="crop", normalize=None)
    dataset = JUMPCPDataset(test_file, img_path,mapping_path,transforms)
    

    
    for fold_id in range(cv_folds):
        all_predictions = []
        all_labels = []
        all_metadata = []
        train_idx, _, test_idx = create_datasplits(data_args, seed_id=SEED+fold_id)
        use_running = args.method != "armbn" and args.method != "armcml"
        model, arm_model = load_model(args, model_path[fold_id], device, model_name, use_running=use_running)
        test_data = CustomSubset(dataset, test_idx)

        if args.method == "tent":
            model.train()
            model.requires_grad_(False)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
            parameters = []
            for namem, m in model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    for namep, p in m.named_parameters():
                        if namep in ['weight', 'bias']:  # weight is scale, bias is shift
                            parameters.append(p)



        if args.method == "memo":
            if args.subset:                     #for hyperparameter search
                np.random.seed(42)
                subset = np.random.choice(len(test_idx), size=450, replace=False)
                test_data = CustomSubset(dataset, subset)
            if args.prior_strength>=0:
                nn.BatchNorm2d.prior = float(args.prior_strength) / float(args.prior_strength + 1)
                nn.BatchNorm2d.forward = _modified_bn_forward
                
        """
        #for precomputing statistics
        if args.method == "memo":
            print("Computing stats")
            train_data = CustomSubset(dataset, train_idx)
            stats = compute_stats(train_data)
            print("Computed stats")
            print(stats)
            continue"""

        if args.method=="erm" or args.method=="memo" or args.method=="tent":
            dataloader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False
        )
        elif "arm" in args.method:
            sampler = GroupSampler(test_data,meta_batch_size=args.meta_batch_size_eval,support_size=args.support_size_eval,uniform_over_groups=False)
            dataloader = DataLoader(
                test_data,
                num_workers=1,
                pin_memory=True,
                batch_sampler=sampler,

            )
        
        for batch in tqdm(dataloader):
            imgs, labels,_, metadata = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            if args.method == "armll":
                inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
                predictions, loss = predict_and_loss(model,imgs,args,labels,loss_fn, is_train=False, arm_net=arm_model, inner_opt=inner_opt)
            elif args.method == "armcml":
                predictions, loss = predict_and_loss(model,imgs,args,labels,loss_fn, is_train=False, arm_net=arm_model)
            elif args.method == "tent":
                inner_opt = torch.optim.SGD(parameters, lr=args.tent_lr, momentum=args.tent_momentum)
                predictions, loss = predict_and_loss(model,imgs,args,labels,loss_fn, is_train=False, inner_opt=inner_opt)
            elif args.method == "memo":
                if args.memo_opt=="SGD": 
                    memo_opt = torch.optim.SGD(model.parameters(), lr=args.memo_lr, weight_decay=args.memo_wd)
                elif args.memo_opt=="AdamW": 
                    memo_opt = torch.optim.AdamW(model.parameters(), lr=args.memo_lr, weight_decay=args.memo_wd)
                predictions, loss = predict_and_loss(model,imgs,args,labels,loss_fn, is_train=False, inner_opt=memo_opt, fold_id =fold_id)
            else:
                predictions, loss = predict_and_loss(model,imgs,args,labels,loss_fn, is_train=False)


            all_predictions.append(predictions)
            all_labels.append(labels)
            all_metadata.append(metadata)

        flat_metadata = []
        for batch_meta in all_metadata:  # each is a dict of lists
            (batch_key, batch_items), (source_key, source_items) = batch_meta.items()
            for i in range(len(batch_items)):
                flat_metadata.append({batch_key: batch_items[i], source_key: source_items[i]})
        
        data.append([torch.cat(all_predictions), torch.cat(all_labels), pd.DataFrame(flat_metadata)])

    return data



def compute_stats(dataset):
    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )
    n_pixels = 0
    channel_sum = torch.zeros(5, device="cuda")
    channel_sq_sum = torch.zeros(5, device="cuda")
    print(len(loader))
    for i, (imgs, _, _, _) in enumerate(loader):
        imgs = imgs.to("cuda", non_blocking=True)
        if i%10 == 0:
            print(i)
        b, c, h, w = imgs.shape
        n_pixels += b * h * w

        channel_sum += imgs.sum(dim=[0, 2, 3])
        channel_sq_sum += (imgs ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std = (channel_sq_sum / n_pixels - mean ** 2).sqrt()

    return mean.cpu(), std.cpu()

def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)




def compute_metrics(data, id_mapping, verbose=True):
    accuracies = []
    for pred, labels, metadata in data:
        pred_labels = torch.argmax(pred, dim=1).cpu()
        correct = (pred_labels == labels.cpu()).sum().item()
        acc = correct/len(labels)
        accuracies.append(acc)
    if verbose:
        print(f"Individual acc: {accuracies}")
        print(f"Test set: \n Mean: {np.mean(accuracies)}\n Std: {np.std(accuracies)}")
        print("\n Accuracy per label")
    
    #compute accuracy per label    
    label_accs = []
    for i, (pred, labels, metadata) in enumerate(data):
        if verbose: print(f"Fold {i+1}")
        label_acc = {}
        collect = []
        labels = labels.cpu()
        for i in labels.unique():
            filter_pred = pred[labels==i]
            filter_pred = torch.argmax(filter_pred, dim=1).cpu()
            correct = (filter_pred == i).sum().item()
            acc = correct/len(filter_pred)

            key = id_mapping[str(i.item())]
            if key not in label_acc:
                label_acc[key] = []
            label_acc[key].append(acc)

        for key, value in label_acc.items():
            collect.append(np.mean(value))
            if verbose: print(f"{key}: {np.mean(value)}")
        label_accs.append(collect)
    
    return accuracies, label_accs
    

