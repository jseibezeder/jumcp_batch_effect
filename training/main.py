from training.parameters import parse_args
from training.model import ResNet, MLP, ContextNet
from training.data import get_data, create_datasplits
from training.train import train, evaluate
from training.logger import setup_primary_logging, setup_worker_logging
from training.scheduler import get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_warm_restarts_schedule_with_warmup

import random
import numpy as np
import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import json
from time import gmtime, strftime
import copy


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

#check if at gpu 0
def is_master(args):
    return (not args.distributed) or args.gpu == 0


def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)

    # Log and save params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)

    model_config_file = Path(__file__).parent / "model_parameter"/ f"{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)

    if args.method == "armcml":
        model_info["input_shape"] += args.n_context_channels

    #issue on how we predict and backpropagate, since when predicting we predict per sub meta-batch
    #but the backward step failed, since we had differnent versions of the batch statistics
    #TODO: check if really set to false for armcml
    if args.method == "armbn" or (args.method == "armcml" and args.adapt_bn):
        model_info["use_batch_running"] = False

    model = ResNet(**model_info)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)
    else:
        model = model.half()


    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    #learned loss model for armll
    inner_opt = None
    arm_net = None
    if args.method == "armll":
        num_classes = model_info["output_dim"]
        device  ="cuda" if torch.cuda.is_available() else "cpu"
        arm_net = MLP(in_size=num_classes, norm_reduce=True).to(device)
        inner_opt = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
        
    elif args.method == "armcml":
        num_classes = model_info["output_dim"]
        device  ="cuda" if torch.cuda.is_available() else "cpu"
        arm_net = ContextNet(5, args.n_context_channels,
                                 hidden_dim=args.cml_hidden_dim, kernel_size=5, use_running_stats=args.adapt_bn).to(device)
    
    if arm_net!=None:
        if args.precision == "fp32" or args.gpu is None:
            convert_models_to_fp32(arm_net)
        else:
            arm_net = arm_net.half()
        if args.distributed and args.use_bn_sync:
            arm_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(arm_net)
        if args.distributed:
            arm_net = torch.nn.parallel.DistributedDataParallel(arm_net, device_ids=[args.gpu])
        
        
    if "arm" in args.method:
        if args.batch_size % args.meta_batch_size_train != 0 and args.batch_size_eval % args.meta_batch_size_eval != 0:
            raise ValueError("batch_size must be divisible by meta_batch_size")
        else:
            args.support_size_train = int(args.batch_size / args.meta_batch_size_train)
            args.support_size_eval = int(args.batch_size_eval / args.meta_batch_size_eval)

    print("Before data")
    data = get_data(args)
    print("Data loaded")


    def exclude(n):
        return "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n):
        return not exclude(n)

    #get specific parameters for different weightdecay
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_file is None:
        optimizer = None
        scheduler = None
    else:
        #define optimizer
        params = [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ]
        if args.method == "armll":
            params.append({"params": arm_net.parameters()}) 

        #define optimizer
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )


        #define total steps
        steps_per_epoch = data["train"].dataloader.num_batches
        total_steps = data["train"].dataloader.num_batches * args.epochs / args.grad_acc
        #get learning rate scheduler
        scheduler = None
        if args.lr_scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup=args.warmup, num_training_steps=total_steps)
        elif args.lr_scheduler == "cosine-restarts":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, warmup=args.warmup,
                                                                           num_cycles=args.restart_cycles,
                                                                           num_training_steps=total_steps)
        elif args.lr_scheduler == "cosine-warm":
            scheduler = get_cosine_with_warm_restarts_schedule_with_warmup(optimizer, warmup=args.warmup,
                                                                           start_restarts=steps_per_epoch*args.start_restart,
                                                                           restarts_multiplication=args.restart_mul)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.save_logs = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and (
            (not args.distributed) or args.gpu == 0
    )
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)


    #if we have a validation set
    if args.train_file is None:
        evaluate(model, data, start_epoch, args, writer, 0, arm_net=arm_net, inner_opt=inner_opt)
        return
    #else training
    elif start_epoch == 0 and  args.val_indices is not None:
        print("Evaluating")
        evaluate(model, data, 0, args, writer, 0, arm_net=arm_net, inner_opt=inner_opt)
    # print("Before train")
    if is_master(args) or not args.distributed:
        prev_best_loss = np.inf
        counter = 0

    for epoch in range(start_epoch, args.epochs):
        if args.gpu == 0:
            logging.info(f'Start epoch {epoch}')


        train(model, data, epoch, optimizer, scheduler, args, writer, arm_net=arm_net, inner_opt=inner_opt)
        steps = data["train"].dataloader.num_batches * (epoch + 1)
        if args.val_indices:
            loss = evaluate(model, data, epoch + 1, args, writer, steps, arm_net=arm_net, inner_opt=inner_opt)

        if args.distributed:
            device = torch.device("cuda", args.gpu) if torch.cuda.is_available() else torch.device("cpu")
            early_stop = torch.zeros(1).to(device)
        else:
            early_stop = 0
        if is_master(args):
            if loss < prev_best_loss:
                prev_best_loss = loss
                counter = 0
            elif loss != np.inf:
                counter += 1

            if counter > args.patience:
                early_stop += 1


        # Saving checkpoints.
        if args.save_logs and (args.gpu == 0 or (not args.distributed)) and (counter == 0 or early_stop==1):
            params_to_save = {
                    "epoch": epoch + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
            if args.method == "armll" or args.method == "armcml":
                params_to_save["armnet_state_dict"] = arm_net.state_dict()
            torch.save(
                params_to_save,
                os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"),
            )



        if args.distributed:
            dist.barrier()
            dist.all_reduce(early_stop, op=dist.ReduceOp.SUM)

        if early_stop == 1:
            logging.info("Stopped early")
            break

def main():
    args = parse_args()
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #check if model configs exist
    model_config_file = Path(__file__).parent / f"model_parameter/{args.model.replace('/', '-')}.json"
    assert os.path.exists(model_config_file)

    img_res_str = str(args.image_resolution_train)

    #check if gpu should be used/cuda is available, and if we should use distrubuted learning
    args.distributed = (args.gpu is None) and torch.cuda.is_available()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    # get the name of the experiments
    if args.name is None:
        args.name = strftime(
            f"method={args.method}_"
            f"imgres={img_res_str}_"
            f"lr={args.lr}_"
            f"wd={args.wd}_"
            f"agg={args.aggregate}_"
            f"model={args.model}_"
            f"world_size={args.world_size}"
            f"batchsize={args.batch_size}_workers={args.workers}_date=%Y-%m-%d-%H-%M-%S_",
            gmtime()
        )
        if args.debug_run:
            args.name += '_DEBUG'

    #use for logging
    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['fp16', 'fp32']

    torch.multiprocessing.set_start_method("spawn", force=True)


    for fold_id in range(args.cross_validation):
        print(f"Start fold {fold_id+1}/{args.cross_validation}")

        fold_args = copy.deepcopy(args)
        fold_args.fold_id = fold_id

        train_idx, val_idx, test_idx = create_datasplits(args, seed_id=args.seed+fold_id)

        fold_args.train_indices = train_idx
        fold_args.val_indices = val_idx
        fold_args.test_indices = test_idx 

        fold_args.name = f"{args.name}/fold{fold_id}" if args.cross_validation > 1 else args.name
        fold_args.log_path = os.path.join(args.logs, fold_args.name, "out.log")
        fold_args.tensorboard_path = os.path.join(args.logs, fold_args.name, "tensorboard") if fold_args.tensorboard else ''
        fold_args.checkpoint_path = os.path.join(args.logs, fold_args.name, "checkpoints")
        for dirname in [fold_args.tensorboard_path, fold_args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)

        fold_args.log_level = logging.DEBUG if fold_args.debug or fold_args.debug_run else logging.INFO
        log_queue = setup_primary_logging(fold_args.log_path, fold_args.log_level)

        if fold_args.distributed:
            ngpus_per_node = torch.cuda.device_count()
            fold_args.world_size = ngpus_per_node
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, log_queue, fold_args))
        else:
            fold_args.world_size = 1
            main_worker(fold_args.gpu, None, log_queue, fold_args)

        torch.cuda.empty_cache()

        print(f"Logging to {fold_args.log_path}")


if __name__ == "__main__":
    main()