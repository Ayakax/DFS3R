# --------------------------------------------------------
# Modified from SimMIM (https://github.com/microsoft/SimMIM)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# --------------------------------------------------------

from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 16
_C.DATA.PIN_MEMORY = False
_C.DATA.NUM_WORKERS = 4
_C.DATA.NUM_CLASS = 1

_C.DATA.IMG_SIZE = 512
_C.DATA.INPUT_CHANNEL = 30
_C.DATA.MASK_PATCH_SIZE = 32
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'DeformableS3RForMIM'
_C.MODEL.BACKBONE = 'resnet18'
_C.MODEL.PRETRAIN = False
_C.MODEL.MODE = 'segmentation'
 
_C.MODEL.DIM = 256
_C.MODEL.HEAD = 8
_C.MODEL.DROPOUT = 0.1
_C.MODEL.NORM_BEFORE = False
_C.MODEL.ACTIVATION = 'relu'
_C.MODEL.NUM_DECODER = 6

# UNet
_C.MODEL.UNET = CN()
_C.MODEL.UNET.ENCODER_CHANNELS = (3, 64, 64, 128, 256, 256)
_C.MODEL.UNET.DECODER_CHANNELS = (256, 128, 64, 32, 16)
_C.MODEL.UNET_BLOCKS = 5

# Deformable tes4r
_C.MODEL.NUM_ENCODER_LAYERS = 2
_C.MODEL.NUM_DECODER_LAYERS = 4
_C.MODEL.AUX_LOSS = False
_C.MODEL.NUM_ENCODER_POINTS = 4  # smaller than spectral group number
_C.MODEL.NUM_DECODER_POINTS = 8 * 10  # key number * spectral group number
_C.MODEL.NUM_CR_EXPERTS = 1
_C.MODEL.NUM_BR_EXPERTS = 256  # equal to the resolution of the last feature map, e.g., 16*16
_C.MODEL.FEATURE_CHANNELS = (64, 128, 256, 512)

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 200
_C.TRAIN.SUB_EPOCHS = 100
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.DEVICE_IDS = [0, 1]

# -----------------------------------------------------------------------------
# Learning rate settings
# -----------------------------------------------------------------------------
_C.TRAIN.LR_SCHEDULER = CN()
# 'StepLR' or 'Cosine'
_C.TRAIN.LR_SCHEDULER.NAME = 'Cosine'

# Cosine scheduler
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-6

# StepLR scheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 5
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.8

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc settings
# -----------------------------------------------------------------------------
_C.CHECKPOINTS_PATH = 'checkpoint/pretrain/' + _C.MODEL.NAME
_C.CHECKPOINTS_NAME = _C.MODEL.BACKBONE
# Fixed random seed
_C.SEED = 19981015
_C.DEVICES = 'cuda'


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    return config
