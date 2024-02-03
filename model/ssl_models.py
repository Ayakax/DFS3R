import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.backbones.deformable_transformer import DeformableTransformer
from model.backbones.unet import Unet


class HsiDecomposition(nn.Module):
    def __init__(self, in_ch, bs):
        super(HsiDecomposition, self).__init__()

        self.channel = in_ch
        self.weights = Parameter(torch.zeros(bs, self.channel - 1, 1))
        nn.init.trunc_normal_(self.weights, std=0.02)

    def spectrum_select(self, tensor):
        src_tensor = torch.transpose(tensor, 1, 3)  # (B, C, H, W)

        split_list, gt_list = [], []
        img_list = list(torch.split(src_tensor, 1))  # (1, C, H, W) * B
        ssl_img = src_tensor.clone()

        for i in range(len(img_list)):
            select_idx = random.randint(0, self.channel - 1)
            spectral_list = list(torch.split(img_list[i].squeeze(0), 1))  # (1, H, W) * C

            ssl_img[i, select_idx, :, :] = 0
            gt_list.append(spectral_list[select_idx].unsqueeze(0))  # (1, 1, H, W)
            spectral_list.pop(select_idx)  # (1, H, W) * (C-1)
            
            split_list.append(torch.cat(tuple(spectral_list)).unsqueeze(0))  # (1, C-1, H, W)
        
        split_img = torch.cat(tuple(split_list))  # (B, C-1, H, W)
        gt_img = torch.cat(tuple(gt_list))  # (B, 1, H, W)

        return split_img, gt_img, ssl_img

    def forward(self, x):
        x = torch.transpose(x, 1, 3)  # (B, H, W, C-1)
        beta = self.weights  # (B, C-1, 1)
        x = torch.einsum('bhwc, bcd -> bhwd', x, beta)  # (B, H, W, 1)
        restore = torch.transpose(x, 1, 3)  # (B, 1, H, W)

        return restore, beta


class DeformableS3RForMIM(nn.Module):
    def __init__(self, config):
        super(DeformableS3RForMIM, self).__init__()
        self.in_c = config.DATA.INPUT_CHANNEL
        self.mask_patch_size = config.DATA.MASK_PATCH_SIZE

        self.encoder, self.upsampler, self.rec_head = self.build_unet(config)
        self.alignment = self.build_alignment(config)
        self.ch_pos = Parameter(torch.zeros(config.DATA.INPUT_CHANNEL, config.MODEL.DIM))
        nn.init.trunc_normal_(self.ch_pos, std=0.02)

        self.decoder = self.build_decoder(config)

        self.mlp = nn.Sequential(
                nn.Linear(config.MODEL.DIM, 2 * config.MODEL.DIM),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(2 * config.MODEL.DIM, self.in_c - 1)
            )

    def build_unet(self, config):
        unet = Unet(
            encoder_name='resnet18',
            encoder_depth=config.MODEL.UNET_BLOCKS,
            encoder_weights="imagenet",
            encoder_channels=config.MODEL.UNET.ENCODER_CHANNELS,
            decoder_channels=config.MODEL.UNET.DECODER_CHANNELS,
            in_channels=3,
            classes=1             
        )

        encoder = unet.encoder
        upsampler = unet.decoder
        head = unet.segmentation_head

        return encoder, upsampler, head

    def build_basic_layer(self, lvl, in_ch, out_ch):
        basic_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((8 * (4 - lvl), 8 * (4 - lvl))),
                nn.Conv2d(in_ch, out_ch, (1, 1)),
                nn.ReLU()
            )

        return basic_layer

    def build_alignment(self, config):
        alignment = nn.ModuleList([self.build_basic_layer(i, config.MODEL.FEATURE_CHANNELS[i],
                                                          config.MODEL.DIM) for i in range(4)])

        return alignment

    def build_decoder(self, config):
        decoder = DeformableTransformer(
            d_model=config.MODEL.DIM,
            nhead=config.MODEL.HEAD,
            num_encoder_layers=config.MODEL.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.MODEL.NUM_DECODER_LAYERS,
            dim_feedforward=2*config.MODEL.DIM,
            dropout=config.MODEL.DROPOUT,
            activation="relu",
            return_intermediate_dec=config.MODEL.AUX_LOSS,
            num_feature_levels=4,
            dec_n_points=config.MODEL.NUM_DECODER_POINTS,
            enc_n_points=config.MODEL.NUM_ENCODER_POINTS,
            num_cr_experts=config.MODEL.NUM_CR_EXPERTS, 
            num_br_experts=config.MODEL.NUM_BR_EXPERTS)

        return decoder

    def patchify(self, imgs):
        """
        imgs: (N, C-1, H, W)
        x: (N*(C-1), L, patch_size**2)
        """
        B = imgs.shape[0] 
        p = self.mask_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(B, self.in_c, h, p, w, p))
        x = torch.einsum('nchpwq->nchwpq', x)
        x = x.reshape(shape=(B * self.in_c, h * w, p ** 2))
        return x

    def unpatchify(self, x):
        """
        x: (N*(C-1), L, patch_size**2)
        imgs: (N*(C-1), 1, H, W)
        """
        p = self.mask_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p))
        x = torch.einsum('nhwpq->nhpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask):
        B, C_ = x.shape[:2]
        ch_idx = torch.arange(0, B * C_)
        mask_idx = mask.view(B * C_, -1)

        patches = self.patchify(x)  # (B*(C-1), N*N, P*P)
        patches[(ch_idx[:, None], mask_idx)] = 0
        x_ = self.unpatchify(patches)
        return x_

    def with_position_embedding(self, z, pos):
        pos_embed = self.ch_pos[pos]  # (B, C-1, 256)
        z = z + pos_embed[:, :, None, :].expand(-1, -1, z.shape[2], -1)  # (B, C-1, N, 256)

        return z
    
    def forward(self, x, mask):
        B, C_, H, W = x.shape
        x_ = self.random_masking(x, mask)  # (B*(C-1), 1, H, W)
        C_ = C_ // 3
        x_ = x_.view(B * C_, 3, H, W)
        feature_list = self.encoder(x_)
        
        # prepare input for encoder
        src_flatten = []
        spatial_shapes = []
        for lvl, sub_alignment in enumerate(self.alignment):
            src = sub_alignment(feature_list[-4:][lvl])  # (B*(C-1), 256, 8, 8)
            h, w = src.shape[-2:]
            spatial_shape = (h * C_, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1).view(B, C_, -1, 256)  # (B, C-1, N, 256)
        # src_flatten = self.with_position_embedding(src_flatten, position)  # (B, C-1, N, 256)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # (4, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  # (4)
        valid_ratios = torch.ones([B, spatial_shapes.shape[0], 2], device=src_flatten.device)  #(B, 4, 2)

        cr_experts, br_experts = self.decoder(src_flatten, spatial_shapes, level_start_index, valid_ratios)

        # coefficient regression
        beta = self.mlp(cr_experts.flatten(1))  # (B, C-1)

        # band regression
        H_ = W_ = int(br_experts.shape[1] ** 0.5)
        br_experts = br_experts.view(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()  # (B, 256, 16, 16)

        spatial_features = []
        for feature in feature_list[:-1]:
            c, h, w = feature.shape[-3:]
            feature = feature.view(B, -1, c, h, w)
            feature = torch.mean(feature, dim=1, keepdim=False)
            spatial_features.append(feature)
        spatial_features.append(br_experts)

        band_ft = self.upsampler(*spatial_features) # (B, 32, 512, 512)
        band_ft = nn.functional.interpolate(
                    band_ft, size=(H, W),
                    mode="bilinear", align_corners=False
                )
        band_rec = self.rec_head(band_ft) # (B, 1, 512, 512)

        return beta, band_rec
    

def build_model(config):
    print("Init Model: {}\nBackbone: {}".format(config.MODEL.NAME, config.MODEL.BACKBONE))
    if config.MODEL.NAME == 'DeformableS3RForMIM':
        model = DeformableS3RForMIM(config)

    return model

