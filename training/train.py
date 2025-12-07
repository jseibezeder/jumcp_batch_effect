import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
import math
import os
import gc
from pathlib import Path
import json
import time
import logging
import higher
import numpy as np
from copy import deepcopy
from torch.nn.functional import softmax
from training.augmentations import augment_and_mix



def is_master(args):
    return (not args.distributed) or args.gpu == 0

#loss for memo
def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

def predict_and_loss(model, x, args, labels, loss_fn, is_train=True, learned_loss_model=None, scaler=None, optimizer=None):
    if args.method == "standard":
        if not is_train:
            model.eval()
            with torch.no_grad():
                pred = model(x)
        else:
            pred = model(x)
        loss = loss_fn(pred, labels)
    
        return pred, loss
    
    elif args.method == "armbn":
        model.train()

        n_domains = math.ceil(len(x) / args.support_size)
        logits = []
        for domain_id in range(n_domains):
            start = domain_id * args.support_size
            end = start + args.support_size
            end = min(len(x), end) # in case final domain has fewer than support size samples
            domain_x = x[start:end]
            if is_train:
                domain_logits = model(domain_x)
            else: 
                with torch.no_grad():
                    domain_logits = model(domain_x)
            logits.append(domain_logits)

        logits = torch.cat(logits)

        loss = loss_fn(logits, labels)

        return logits, loss
    
    elif args.method == "armll":
        model.train()
        learned_loss_model.train()
        if optimizer:
            optimizer.zero_grad(set_to_none=True)

        n_domains = math.ceil(len(x) / args.support_size)
        with autocast(cache_enabled=True):
            logits = []
            loss = []
            base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            base_model.train()
            inner_opt = torch.optim.SGD(base_model.parameters(), lr=args.inner_lr)
            for domain_id in range(n_domains):
                start = domain_id*args.support_size
                end = start + args.support_size
                end = min(len(x), end) # in case final domain has fewer than support size samples

                domain_x = x[start:end]

                with higher.innerloop_ctx(
                    base_model, inner_opt, copy_initial_weights=False, track_higher_grads=True) as (fnet, diffopt):
                    # Inner loop
                    for _ in range(args.n_inner_iter):
                        #with autocast():
                        spt_logits = fnet(domain_x)

                        spt_loss = learned_loss_model(spt_logits)

                        diffopt.step(spt_loss)

                    # Evaluate
                    
                    domain_logits = fnet(domain_x)
                    logits.append(domain_logits.detach().cpu())

                    domain_labels = labels[start:end]
                    domain_loss = loss_fn(domain_logits, domain_labels)
                    if is_train and labels is not None:
                        if scaler:
                            scaler.scale(domain_loss).backward()
                        else:
                            domain_loss.backward()
                    loss.append(domain_loss.to('cpu').detach().item())
                #have to delete, since if not some compuational graphs leak and an OutOfMemory(OOM) error occurs
                del spt_logits, spt_loss, domain_logits, domain_loss
                torch.cuda.empty_cache()
        if scaler:
            scaler.step(optimizer)
            scaler.update()

        logits = torch.cat(logits)


        if is_train and args.precision=="amp":
            optimizer.zero_grad(set_to_none=True)
            return logits, np.mean(loss), scaler, optimizer
        else:
            return logits, np.mean(loss)
        
    elif args.method == "memo":
        if is_train:
            pred = model(x)

        else:
            original_state = deepcopy(model.state_dict())
            
            pred = []

            for x_ind in x:
                model.eval()            
                memo_opt = torch.optim.SGD(model.parameters(), lr=args.memo_lr)

                if args.prior_strength < 0:
                    nn.BatchNorm2d.prior = 1
                else:
                    nn.BatchNorm2d.prior = float(args.prior_strength) / float(args.prior_strength + 1)


                aug_inputs = torch.stack([augment_and_mix(x_ind.clone(), seed=args.seed) for _ in range(args.k_augmentations)], dim=0).requires_grad_()

                for _ in range(args.memo_steps):
                    logits = model(aug_inputs)
                    memo_opt.zero_grad()

                    loss, logits = marginal_entropy(logits)
                    
                    loss.backward()
                    memo_opt.step()
                nn.BatchNorm2d.prior = 1

                if args.prior_strength < 0:
                    nn.BatchNorm2d.prior = 1
                else:
                    nn.BatchNorm2d.prior = float(args.prior_strength) / float(args.prior_strength + 1)
                with torch.no_grad():
                    pred.append(model(x_ind.unsqueeze(0)))
                model.load_state_dict(original_state)
                nn.BatchNorm2d.prior = 1
                
            pred = torch.cat(pred, dim=0)

        loss = loss_fn(pred, labels)
    
        return pred, loss

def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, learned_loss_model = None):
    os.environ["WDS_EPOCH"] = str(epoch)

    gc.collect()
    torch.cuda.empty_cache()



    model.train()
    dataloader, sampler = data['train'].dataloader, data['train'].sampler


    loss_fn = nn.CrossEntropyLoss()

    model_config_file = Path(__file__).parent / f"model_parameter/{args.model.replace('/', '-')}.json"
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)

    if args.gpu is not None:
        loss_fn = loss_fn.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    #TODO: fix percentage
    if args.method == "standard":
        num_batches_per_epoch = dataloader.num_batches
    else:
        num_batches_per_epoch = int(dataloader.num_batches/args.world_size)

    end = time.time()
    #print("Before 1st batch")
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        imgs, labels, _, metadata = batch


        if args.gpu is not None:
            images = imgs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end
        m = model.module if args.distributed else model

        # with automatic mixed precision.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.precision == "amp" and args.method=="armll":
            pred, total_loss,scaler, optimizer = predict_and_loss(model, 
                                                    images, 
                                                    args, 
                                                    loss_fn=loss_fn, 
                                                    labels=labels, 
                                                    learned_loss_model=learned_loss_model, 
                                                    scaler=scaler,
                                                    optimizer=optimizer)
        elif args.precision == "amp" :
            with autocast(cache_enabled=True):
                pred, total_loss = predict_and_loss(model, 
                                                    images, 
                                                    args, 
                                                    loss_fn=loss_fn, 
                                                    labels=labels, 
                                                    learned_loss_model=learned_loss_model, 
                                                    scaler=scaler)
                #torch.autograd.set_detect_anomaly(True)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            pred, total_loss = predict_and_loss(model, 
                                          images, 
                                          args, 
                                          loss_fn=loss_fn, 
                                          labels=labels,
                                          learned_loss_model=learned_loss_model)
            #torch.autograd.set_detect_anomaly(True)
            if args.method != "armll":
                total_loss.backward()
            optimizer.step()

        scheduler.step()

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            log_str = f""
            log_data = {}

            # logging
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}{log_str}"
            )

            # save train loss / etc.
            log_data.update({
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "lr": optimizer.param_groups[0]["lr"]
            })

            # log to tensorboard and/or wandb
            timestep = epoch * num_batches_per_epoch + i
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)


def evaluate(model, data, epoch, args, tb_writer=None, steps=None, learned_loss_model = None):
    if not is_master(args):
        return

    #model.eval()

    dataloader = data['val'].dataloader

    loss_fn = nn.CrossEntropyLoss()
    model_config_file = Path(__file__).parent / f"model_parameter/{args.model.replace('/', '-')}.json"
    with open(model_config_file, 'r') as f:
            model_info = json.load(f)

    if args.gpu is not None:
        loss_fn = loss_fn.cuda(args.gpu)


    cumulative_loss = 0.0
    num_elements = 0.0
    correct = 0.0


    for batch in dataloader:
        images, labels, _, metadata = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        if args.method == "armll":
            features, loss = predict_and_loss(model,images,args, is_train=False, loss_fn=loss_fn, labels=labels, learned_loss_model=learned_loss_model)
        else:
            features, loss = predict_and_loss(model,images,args, is_train=False, loss_fn=loss_fn, labels=labels)
        batch_size = len(images)
        cumulative_loss += loss * batch_size
        num_elements += batch_size

        features = features.to("cpu")
        labels = labels.to("cpu")


        preds = torch.argmax(features, dim=1)
        correct += (preds == labels).sum().item()


    loss = cumulative_loss / num_elements
    acc = correct / num_elements

    logging.info(
        f"Eval Epoch: {epoch}, Loss: {loss}, Accuracy: {acc} "
    )

    if args.save_logs:
        for name, val in {"loss": loss, "accuracy": acc}.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)
    """if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
        f.write(json.dumps(metrics))
        f.write("\n")"""

    return loss

def get_cosine_with_warm_restarts_schedule_with_warmup(
        optimizer: Optimizer, warmup: int, start_restarts:int = 10, restarts_multiplication: int = 2):
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=warmup)

    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=start_restarts, T_mult=restarts_multiplication, eta_min=1e-6)

    return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup])


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, warmup: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        warmup (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, num_training_steps - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer: Optimizer, warmup: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        warmup (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, num_training_steps - warmup))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)