import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import gc
from pathlib import Path
import json
import time
import logging


def is_master(args):
    return (not args.distributed) or args.gpu == 0

def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)

    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    dataloader, sampler = data['train'].dataloader, data['train'].sampler


    loss = nn.CrossEntropyLoss()

    model_config_file = Path(__file__).parent / f"model_parameter/{args.model.replace('/', '-')}.json"
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)

    if args.gpu is not None:
        loss = loss.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    #print("Before 1st batch")
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        imgs, labels = batch


        if args.gpu is not None:
            images = imgs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end
        m = model.module if args.distributed else model

        # with automatic mixed precision.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.precision == "amp":
            with autocast(device_type = device):
                total_loss = loss(model(images), labels)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = loss(model(images), labels)
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


def evaluate(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return

    model.eval()

    dataloader = data['val'].dataloader

    loss = nn.CrossEntropyLoss()
    model_config_file = Path(__file__).parent / f"model_parameter/{args.model.replace('/', '-')}.json"
    with open(model_config_file, 'r') as f:
            model_info = json.load(f)

    if args.gpu is not None:
        loss = loss.cuda(args.gpu)


    cumulative_loss = 0.0
    num_elements = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            features = model(images)
            batch_size = len(images)
            cumulative_loss += loss(features, labels) * batch_size
            num_elements += batch_size


        loss = cumulative_loss / num_elements

        logging.info(
            f"Eval Epoch: {epoch}, Loss: {loss} "
        )


    """if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")"""

    return loss


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