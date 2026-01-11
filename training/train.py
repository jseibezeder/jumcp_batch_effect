import torch
import torch.nn as nn

import math
import os
import gc
import time
import logging
import higher
import numpy as np
from training.augmentations import augment_and_mix, get_aug
from copy import deepcopy



def is_master(args):
    return (not args.distributed) or args.gpu == 0

#loss for memo
def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

#loss for tent
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def predict_and_loss(model, x, args, labels, loss_fn, is_train=True, arm_net=None, inner_opt=None, fold_id = None, ablation=None, use_augmix=True):
    if "arm" in args.method:
        if is_train:
            meta_batch_size = args.meta_batch_size_train
            support_size = args.support_size_train
        else:
            meta_batch_size = args.meta_batch_size_eval
            support_size = args.support_size_eval

    if args.method == "erm":
        if not is_train:
            model.eval()
            with torch.no_grad():
                pred = model(x)
        else:
            model.train()
            pred = model(x)
        loss = loss_fn(pred, labels)
    
        return pred, loss
    

    
    elif args.method == "armcml":
        batch_size, c, h, w = x.shape
        if args.adapt_bn:
            arm_net.train()
            model.train()
            out = []
            for i in range(meta_batch_size):
                x_i = x[i*support_size:(i+1)*support_size]
                if is_train:
                    context_i = arm_net(x_i)
                else:
                    with torch.no_grad():
                        context_i = arm_net(x_i)
                context_i = context_i.mean(dim=0).expand(support_size, -1, -1, -1)
                x_i = torch.cat([x_i, context_i], dim=1)
                out.append(model(x_i))
            logits = torch.cat(out)
        else:
            if is_train:
                arm_net.train()
                model.train()
            else:
                arm_net.eval()
                model.eval()
            if is_train:
                context = arm_net(x)
            else:
                with torch.no_grad():
                    context = arm_net(x)
            context = context.reshape((meta_batch_size, support_size, args.n_context_channels, h, w))
            context = context.mean(dim=1) 
            context = torch.repeat_interleave(context, repeats=support_size, dim=0)
            x = torch.cat([x, context], dim=1)
            if is_train:
                logits = model(x)
            else:
                with torch.no_grad():
                    logits = model(x)
        if is_train:
            loss = loss_fn(logits, labels)
        else:
            with torch.no_grad():
                loss = loss_fn(logits, labels)
        return logits.detach().cpu(), loss




    elif args.method == "armbn":
        model.train()

        n_domains = math.ceil(len(x) / support_size)
        logits = []
        for domain_id in range(n_domains):
            start = domain_id * support_size
            end = start + support_size
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

        n_domains = math.ceil(len(x) / support_size)
        logits = []
        loss = []
        base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for domain_id in range(n_domains):
            start = domain_id*support_size
            end = start + support_size
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
                logits.append(domain_logits.detach().cpu())
                domain_labels = labels[start:end]
                domain_loss = loss_fn(domain_logits, domain_labels)
                if is_train and labels is not None:
                    domain_loss.backward()
                loss.append(domain_loss.to('cpu').detach().item())


        logits = torch.cat(logits)

        return logits, np.mean(loss)
        




    elif args.method == "memo":
        if is_train:
            pred = model(x)

        else:
            #orig_params = [p.detach().clone() for p in model.parameters()]
            model_state = deepcopy(model.state_dict())
            pred = []
            
            if args.prior_strength < 0:
                nn.BatchNorm2d.prior = 1
            else:
                nn.BatchNorm2d.prior = float(args.prior_strength) / float(args.prior_strength + 1)

            for x_ind in x:
                model.eval()           
                aug = get_aug()
                if use_augmix:
                    aug_inputs = torch.stack([augment_and_mix(x_ind, severity=args.severity, fold_id =fold_id, ablation=ablation) for _ in range(args.k_augmentations)], dim=0).requires_grad_()
                else:
                    aug_inputs = torch.stack([aug(x_ind) for _ in range(args.k_augmentations)], dim=0).requires_grad_()
                for _ in range(args.memo_steps):
                    logits = model(aug_inputs)
                    inner_opt.zero_grad()

                    loss, logits = marginal_entropy(logits)
                    
                    loss.backward()
                    inner_opt.step()

                with torch.no_grad():
                    pred.append(model(x_ind.unsqueeze(0)))
                model.load_state_dict(model_state, strict=True)
            
            nn.BatchNorm2d.prior = 1
                
            pred = torch.cat(pred, dim=0)

        loss = loss_fn(pred, labels)
    
        return pred, loss
    elif args.method == "tent":
        if is_train:
            pred = model(x)
        else:
            if args.episodic:
                model_state = deepcopy(model.state_dict())
                inner_opt_state = deepcopy(inner_opt.state_dict())
            for _ in range(args.tent_steps):
                pred = model(x)
                loss = softmax_entropy(pred).mean(0)
                loss.backward()
                inner_opt.step()
                inner_opt.zero_grad()
            if args.episodic:
                model.load_state_dict(model_state, strict=True)
                inner_opt.load_state_dict(inner_opt_state)
            
        loss = loss_fn(pred, labels)

        return pred, loss




def train(model, data, epoch, optimizer, scheduler, args, tb_writer=None, arm_net = None, inner_opt=None):
    os.environ["WDS_EPOCH"] = str(epoch)

    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    dataloader, sampler = data['train'].dataloader, data['train'].sampler


    loss_fn = nn.CrossEntropyLoss()


    if args.gpu is not None:
        loss_fn = loss_fn.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):


        imgs, labels, _, metadata = batch
        

        if args.gpu is not None:
            images = imgs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)


        data_time = time.time() - end
    
        pred, total_loss = predict_and_loss(model, 
                                        images, 
                                        args, 
                                        loss_fn=loss_fn, 
                                        labels=labels,
                                        arm_net=arm_net,
                                        inner_opt=inner_opt)
        if args.method != "armll":
            total_loss.backward()

        if (i+1)%args.grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
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


def evaluate(model, data, epoch, args, tb_writer=None, steps=None, arm_net = None, inner_opt=None):
    
    if not is_master(args):
        return 

    dataloader = data['val'].dataloader

    loss_fn = nn.CrossEntropyLoss()

    if args.gpu is not None:
        loss_fn = loss_fn.cuda(args.gpu)


    cumulative_loss = 0.0
    num_elements = 0.0
    correct = 0.0

    #need to use the base model without the wrapper or elsewise early stopping wont work
    base_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    base_arm_net = arm_net.module if isinstance(arm_net, torch.nn.parallel.DistributedDataParallel) else arm_net


    for batch in dataloader:
        images, labels, _, metadata = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        features, loss = predict_and_loss(base_model,images,args, is_train=False, loss_fn=loss_fn, labels=labels, arm_net=base_arm_net,inner_opt=inner_opt)

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
  
    return loss

