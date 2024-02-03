import os
import warnings
from tqdm import tqdm

import torch
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F

from utils import build_scheduler, fix_random_seeds
from dataset.ssl_dataset import build_loader
from model.ssl_models import build_model, HsiDecomposition
from config.ssl_config import get_config
 
warnings.filterwarnings('ignore')


def train_one_epoch(epoch, config, model, train_loader, optimizer):
    model.train()
    train_loss, avg_cr_loss, avg_br_loss = 0., 0., 0.

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for batch_idx, (data, mask) in pbar:
        dcom = HsiDecomposition(in_ch=config.DATA.INPUT_CHANNEL, bs=data.shape[0])
        src_dcom, target, src_ssl = dcom.spectrum_select(data)

        dcom_opt = optim.AdamW(dcom.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-2)
        dcom_scheduler = optim.lr_scheduler.StepLR(dcom_opt, step_size=1, gamma=0.7)
        dcom.loss_func = torch.nn.L1Loss()
        dcom.train()

        if config.DEVICES == 'cuda':
            src_dcom, target = src_dcom.cuda(), target.cuda()
            dcom.cuda()

        # print('\n', '*' * 25, 'HSI Decomposition', '*' * 25)
        for sub_epochs in range(1, config.TRAIN.SUB_EPOCHS + 1):
            dcom_opt.zero_grad()
            restore, beta = dcom.forward(src_dcom)
            dcom_loss = dcom.loss_func(restore, target)
            dcom_loss.backward()
            dcom_opt.step()

            if sub_epochs % 200 == 0:
                # print('Sub Epoch: {}, Loss: {:.4f}'.format(sub_epochs, dcom_loss.item()))
                dcom_scheduler.step()

        beta = beta.view(restore.shape[0], -1)

        optimizer.zero_grad()
        beta_rec, band_rec = model.forward(src_ssl, mask)
        
        # loss
        cr_loss = F.l1_loss(beta_rec, beta)
        br_loss = F.l1_loss(band_rec, target)
        loss = cr_loss + br_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * beta.shape[0] / train_loader.dataset.size
        avg_cr_loss += cr_loss.item() * beta.shape[0] / train_loader.dataset.size
        avg_br_loss += br_loss.item() * beta.shape[0] / train_loader.dataset.size

        if config.TRAIN.LR_SCHEDULER.NAME == 'Cosine':
            lr_scheduler.step_update(epoch * len(train_loader) + batch_idx)

    print('\nEpoch: {}, Loss: {:.4f}, CRLoss: {:.4f}, BRLoss: {:.4f}'.format(epoch,
            train_loss, avg_cr_loss, avg_br_loss))


if __name__ == "__main__":
    # Training settings
    config = get_config()

    fix_random_seeds(config.SEED)
    device = torch.device(config.DEVICES)

    train_loader = build_loader(config)

    print('Init Model')
    model = build_model(config)
    model.to(device)
    model = torch.nn.DataParallel(module=model, device_ids=config.TRAIN.DEVICE_IDS)

    optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.BASE_LR,
                            betas=config.TRAIN.OPTIMIZER.BETAS, weight_decay=config.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_scheduler(optimizer, config, train_loader)

    print('Start Training')
    os.makedirs(config.CHECKPOINTS_PATH, exist_ok=True)
    for epoch in range(1, config.TRAIN.EPOCHS + 1):
        train_one_epoch(epoch, config, model, train_loader, optimizer)
        if config.TRAIN.LR_SCHEDULER.NAME == 'StepLR':
            lr_scheduler.step()
        
        if epoch % 10 == 0:
            save_path = f"{config.CHECKPOINTS_PATH}/{config.CHECKPOINTS_NAME}_{epoch}.pth"
            torch.save(model.module.state_dict(), save_path)
