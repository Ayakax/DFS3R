# coding:utf-8
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MaskGenerator:
    def __init__(self, input_size=[512, 512], input_channel=32, 
                 model_patch_size=32, mask_ratio=0.6):
        self.input_size = input_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.input_channel = input_channel

        self.token_per_channel = (self.input_size[0] // self.model_patch_size) * (self.input_size[1] // self.model_patch_size)
        self.mask_count = int(np.ceil(self.token_per_channel * self.mask_ratio))

    def __call__(self):
        mask_idx_list = [np.random.permutation(self.token_per_channel)[np.newaxis, :]
                            for i in range(self.input_channel)]
        mask = np.concatenate(mask_idx_list, axis=0)[:, :self.mask_count]  # (C, L*mask_ratio)
            
        return mask


class HsiDataset(Dataset):
    def __init__(self, file_name, config):
        self.file_name = file_name
        self.size = 0
        self.img_list = []

        self.mask_generator = MaskGenerator(
            input_size=[config.DATA.IMG_SIZE, config.DATA.IMG_SIZE],
            input_channel=config.DATA.INPUT_CHANNEL,
            model_patch_size=config.DATA.MASK_PATCH_SIZE,
            mask_ratio=config.DATA.MASK_RATIO
        )

        if not os.path.isfile(self.file_name):
            print(self.file_name + 'does not exist!')
        file = open(self.file_name)
        for f in file:
            self.img_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.img_list[idx].split(' ')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None

        img = np.load(image_path)[..., 1:-1]
        mask = self.mask_generator().astype('int64')

        return img, mask


def build_loader(config):
    train_set = HsiDataset(file_name='./dataset/segmentation/train.txt', config=config)
    print('Load Training Set') 
    
    train_loader = DataLoader(dataset=train_set,
                              batch_size=config.DATA.BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.DATA.NUM_WORKERS,
                              pin_memory=config.DATA.PIN_MEMORY)

    return train_loader
