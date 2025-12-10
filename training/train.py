import torch
import torch.nn as nn

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

def predict_and_loss(model, x, args, labels, loss_fn, is_train=True, arm_net=None):
    if args.method == "erm":
        if not is_train:
            model.eval()
            with torch.no_grad():
                pred = model(x)
        else:
            pred = model(x)
        loss = loss_fn(pred, labels)
    
        return pred, loss
    
    elif args.method == "armcml":
        batch_size, c, h, w = x.shape
        torch.autograd.set_detect_anomaly(True)
        if args.adapt_bn:
            out = []
            for i in range(args.meta_batch_size):
                x_i = x[i*args.support_size:(i+1)*args.support_size]
                context_i = arm_net(x_i)
                context_i = context_i.mean(dim=0).expand(args.support_size, -1, -1, -1)
                x_i = torch.cat([x_i, context_i], dim=1)
                out.append(model(x_i))
            logits = torch.cat(out)
        else:
            context = arm_net(x) # Shape: batch_size, channels, H, W
            context = context.reshape((args.meta_batch_size, args.support_size, args.n_context_channels, h, w))
            context = context.mean(dim=1) # Shape: meta_batch_size, self.n_context_channels
            context = torch.repeat_interleave(context, repeats=args.support_size, dim=0) # meta_batch_size * support_size, context_size
            x = torch.cat([x, context], dim=1)
            logits = model(x)
        
        loss = loss_fn(logits, labels)
        return logits, loss

    
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
        arm_net.train()

        n_domains = math.ceil(len(x) / args.support_size)
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
                base_model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Inner loop
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(domain_x)

                    spt_loss = arm_net(spt_logits)

                    diffopt.step(spt_loss)

                # Evaluate
                
                domain_logits = fnet(domain_x)
                logits.append(domain_logits)
                #if backprop_loss and labels is not None:
                domain_labels = labels[start:end]
                domain_loss = loss_fn(domain_logits, domain_labels)
                if is_train and labels is not None:
                    domain_loss.backward()
                loss.append(domain_loss.to('cpu').detach().item())
            #have to delete, since if not some compuational graphs leak and an OutOfMemory(OOM) error occurs
            #del spt_logits, spt_loss, domain_logits, domain_loss
            #torch.cuda.empty_cache()

        logits = torch.cat(logits)

        #TODO:
        return logits, np.mean(loss)
        
    elif args.method == "memo":
        if is_train:
            pred = model(x)

        else:
            #original_state = deepcopy(model.state_dict())
            orig_params = [p.detach().clone() for p in model.parameters()]
            pred = []

            for x_ind in x:
                model.eval()            
                memo_opt = torch.optim.SGD(model.parameters(), lr=args.memo_lr)

                if args.prior_strength < 0:
                    nn.BatchNorm2d.prior = 1
                else:
                    nn.BatchNorm2d.prior = float(args.prior_strength) / float(args.prior_strength + 1)


                aug_inputs = torch.stack([augment_and_mix(x_ind, seed=args.seed) for _ in range(args.k_augmentations)], dim=0).requires_grad_()
                print(aug_inputs)
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
                #model.load_state_dict(original_state)
                with torch.no_grad():
                    for p, orig in zip(model.parameters(), orig_params):
                        p.copy_(orig)
                nn.BatchNorm2d.prior = 1
                
            pred = torch.cat(pred, dim=0)

        loss = loss_fn(pred, labels)
    
        return pred, loss

def train(model, data, epoch, optimizer, scheduler, args, tb_writer=None, arm_net = None):
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
    if args.method == "erm":
        num_batches_per_epoch = dataloader.num_batches
    else:
        num_batches_per_epoch = int(dataloader.num_batches/args.world_size)

    end = time.time()

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
    
        pred, total_loss = predict_and_loss(model, 
                                        images, 
                                        args, 
                                        loss_fn=loss_fn, 
                                        labels=labels,
                                        arm_net=arm_net)
        #torch.autograd.set_detect_anomaly(True)
        if args.method != "armll":
            total_loss.backward()

        """if args.method == "armll":
            if (i+1)%8==0:
                optimizer.step()
        else:
            optimizer.step()"""
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


def evaluate(model, data, epoch, args, tb_writer=None, steps=None, arm_net = None):
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

        features, loss = predict_and_loss(model,images,args, is_train=False, loss_fn=loss_fn, labels=labels, arm_net=arm_net)

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

