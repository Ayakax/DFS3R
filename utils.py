import os
import torch
import numpy as np
import torch.distributed as dist
from timm.scheduler.cosine_lr import CosineLRScheduler


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    seed = seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_scheduler(optimizer, config, train_loader):
    if config.TRAIN.LR_SCHEDULER.NAME == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS,
            config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    elif config.TRAIN.LR_SCHEDULER.NAME == 'Cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=int(config.TRAIN.EPOCHS * len(train_loader)),
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=int(config.TRAIN.WARMUP_EPOCHS * len(train_loader)),
            cycle_limit=1,
            t_in_epochs=False,
        )

    return lr_scheduler


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=1)
    return dice


class MetricLogger(object):
    def __init__(self, config):
        self.best_metirc = 0
        self.metric_list = []

        self.patience = 0
        self.mode = config.TRAIN.MODE
        self.max_patience = config.TRAIN.PATIENCE
        self.threshold = config.TRAIN.THRESHOLD
        self.stop_flag = False
        self.save_path = f"{config.CHECKPOINTS_PATH}/{config.CHECKPOINTS_NAME}"

    def early_stop(self, metric, model):
        self.patience += 1

        if metric >= self.best_metirc:
            self.best_metirc = metric
            self.patience = 0
            if self.mode == 'ddp' and is_main_process():
                torch.save(model.state_dict(), self.save_path)
            elif self.mode == 'dp':
                torch.save(model.state_dict(), self.save_path)

        if self.patience >= self.max_patience and self.best_metirc > self.threshold:
            self.stop_flag = True

    def cal_metric(self, mode='segmentation'):
        metrics = np.concatenate(self.metric_list)
        self.metric_list = []

        if mode == 'classification':
            return metrics.mean()
        else:
            return metrics.mean(), metrics.std()


def init_distributed_mode(config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.TRAIN.RANK = int(os.environ["RANK"])
        config.TRAIN.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        config.TRAIN.GPU = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        config.TRAIN.RANK = int(os.environ['SLURM_PROCID'])
        config.TRAIN.GPU = config.TRAIN.RANK % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        config.TRAIN.DISTRIBUTED = False
        return

    config.TRAIN.DISTRIBUTED = True

    torch.cuda.set_device(config.TRAIN.GPU)
    config.TRAIN.DIST_BACKEND = 'nccl' 
    config.TRAIN.DIST_URL = 'env://'
    print('| distributed init (rank {}): {}'.format(
        config.TRAIN.RANK, config.TRAIN.DIST_URL), flush=True)
    torch.distributed.init_process_group(backend=config.TRAIN.DIST_BACKEND, init_method=config.TRAIN.DIST_URL,
                                         world_size=config.TRAIN.WORLD_SIZE, rank=config.TRAIN.RANK)
    torch.distributed.barrier()
    setup_for_distributed(config.TRAIN.RANK == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0