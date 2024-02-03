import time
import warnings
from tqdm import tqdm

import torch
from torch.cuda import amp
import torch.optim as optim

from config.seg_config import get_config
from dataset.seg_dataset import build_loader
from model.seg_models import build_model, build_loss
from utils import MetricLogger, build_scheduler, dice_coef, fix_random_seeds
 
warnings.filterwarnings('ignore')


def train_one_epoch(model, train_loader, optimizer, losses_dict, epoch):
    model.train()
    losses_all, bce_all, dice_all = 0, 0, 0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for batch_idx, (images, masks) in pbar:
        optimizer.zero_grad()

        images = images.to(device, dtype=torch.float) # [b, c, w, h]
        masks  = masks.to(device, dtype=torch.float)  # [b, c, w, h]

        y_preds = model(images) # [b, c, w, h]
    
        bce_loss = losses_dict["BCELoss"](y_preds, masks)
        dice_loss = losses_dict["DiceLoss"](y_preds, masks)
        losses = dice_loss

        losses.backward()
        optimizer.step()
        
        losses_all += losses.item() / images.shape[0]
        bce_all += bce_loss.item() / images.shape[0]
        dice_all += dice_loss.item() / images.shape[0]

        if config.TRAIN.LR_SCHEDULER.NAME == 'Cosine':
            lr_scheduler.step_update(epoch * len(train_loader) + batch_idx)
    
    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.6f}".format(current_lr), flush=True)
    print("loss: {:.3f}, bce_all: {:.3f}, dice_all: {:.3f}".format(losses_all, bce_all, dice_all), flush=True)


@torch.no_grad()
def valid_one_epoch(model, valid_loader, logger):
    model.eval()

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, masks) in pbar:
        images  = images.to(device, dtype=torch.float) # [b, c, w, h]
        masks   = masks.to(device, dtype=torch.float)  # [b, c, w, h]
        
        y_preds = model(images)

        batch_dice = dice_coef(masks, y_preds.sigmoid()).detach().cpu().numpy()
        logger.metric_list.append(batch_dice)
    
    val_dice, val_std = logger.cal_metric()
    logger.early_stop(val_dice, model)
    print("val_dice: {:.4f}, val_std: {:.4f}".format(val_dice, val_std), flush=True)


if __name__ == "__main__":
    # Training settings
    config = get_config()
    print(config, '\n')

    fix_random_seeds(config.SEED)
    device = torch.device(config.DEVICES)
        
    train_loader, val_loader, test_loader = build_loader(config)

    print('Init Model')
    model = build_model(config)
    model.to(device)
    model = torch.nn.DataParallel(module=model, device_ids=[0, 1])

    optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.BASE_LR,
                            betas=config.TRAIN.OPTIMIZER.BETAS, weight_decay=config.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_scheduler(optimizer, config, train_loader)
    losses_dict = build_loss()

    logger = MetricLogger(config)

    print('Start Training')
    for epoch in range(1, config.TRAIN.EPOCHS + 1):
        start_time = time.time()
        train_one_epoch(model, train_loader, optimizer, losses_dict, epoch)
        if config.TRAIN.LR_SCHEDULER.NAME == 'StepLR':
            lr_scheduler.step()
        valid_one_epoch(model, val_loader, logger)
        epoch_time = time.time() - start_time
        print("epoch: {}, time: {:.2f}s, best: {:.4f}\n".format(epoch, epoch_time, logger.best_metirc), flush=True)
        
        if logger.stop_flag:
            print('\nEarly Stop')
            break
    
    print('\nStart Testing')
    model.load_state_dict(torch.load(logger.save_path))
    valid_one_epoch(model, test_loader, logger)