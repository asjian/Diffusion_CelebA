import torch
import torch.nn as nn
import math

def train_one_epoch(model, dataloader, optimizer, device, epoch, print_freq = 10, lr_update_freq = 100, lr_scheduler = None, gradscaler = None, grad_clip_norm_max = None, warmup_lr = True):
    model.train()
    loss_avg, loss_n = 0, 0

    warmup_iters = 1000
    if epoch == 0 and warmup_lr:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0/warmup_iters, end_factor=1.0, total_iters=warmup_iters)

    for itr, batch_data in enumerate(dataloader):    
        if epoch == 0 and itr < warmup_iters:
            warmup_scheduler.step()
        
        else:
            if itr%lr_update_freq == 0 and lr_scheduler is not None:
                lr_scheduler.step()

        images = batch_data[0].to(device)

        with torch.amp.autocast(device_type = 'cuda', enabled = (gradscaler is not None)):
            loss = model(images)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            print(loss)
            break

        optimizer.zero_grad()
        if gradscaler is not None:
            gradscaler.scale(loss).backward()

            if grad_clip_norm_max is not None:
                gradscaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_max)
            
            gradscaler.step(optimizer)
            gradscaler.update()

        else:
            loss.backward()
            if grad_clip_norm_max is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm_max)

            optimizer.step()
        
        loss_n += 1
        loss_avg += (loss.item() - loss_avg)/loss_n
        
        if itr%print_freq == 0:
            print(f"Epoch: {epoch}, itr: {itr}, lr: {lr_scheduler.get_last_lr()[0] if lr_scheduler is not None and (epoch > 0 or itr >= warmup_iters) else warmup_scheduler.get_last_lr()[0]} Loss: {loss}, Running Avg Loss: {loss_avg}")