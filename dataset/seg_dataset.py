# coding:utf-8
import os
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader


class HsiDataset(Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        self.size = 0
        self.img_list = []

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
        mask_path = self.img_list[idx].split(' ')[1].split('\n')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None

        img = np.transpose(np.load(image_path)[:, :, 1:-1], (2, 0, 1))
        mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)[np.newaxis, :, :]

        return img, mask / 255.0


def build_dataset():
    train_dataset = HsiDataset(file_name='./dataset/segmentation/train.txt')
    print('Load Training Set')

    test_dataset = HsiDataset(file_name='./dataset/segmentation/test.txt')
    print('Load Test Set')

    validation_dataset = HsiDataset(file_name='./dataset/segmentation/val.txt')
    print('Load Val Set')

    return train_dataset, validation_dataset, test_dataset


def build_loader(config):
    train_set, val_set, test_set = build_dataset()
    
    train_loader = DataLoader(dataset=train_set,
                              batch_size=config.DATA.BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.DATA.NUM_WORKERS,
                              pin_memory=config.DATA.PIN_MEMORY)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=True,
                            num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=config.DATA.PIN_MEMORY)
    
    test_loader = DataLoader(dataset=test_set,
                             batch_size=config.DATA.BATCH_SIZE,
                             shuffle=True,
                             num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=config.DATA.PIN_MEMORY)

    return train_loader, val_loader, test_loader
