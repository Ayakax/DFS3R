import torch
from torch import nn
import segmentation_models_pytorch as smp
from model.ssl_models import DeformableS3RForMIM


class DeformableS3R(nn.Module):
    def __init__(self, config):
        super(DeformableS3R, self).__init__()
        self.mode = config.MODEL.MODE
        self.ssl_model = DeformableS3RForMIM(config)

        if config.MODEL.PRETRAIN:
            print("Load pretrain model")
            self.ssl_model.load_state_dict(torch.load(config.MODEL.PRETRAIN_CKPT))
        self.encoder = self.ssl_model.encoder
        self.alignment = self.ssl_model.alignment
        self.ch_pos = self.ssl_model.ch_pos
        self.decoder = self.ssl_model.decoder

        if self.mode == 'classification':
            self.mlp = nn.Sequential(
                    nn.Linear(config.MODEL.DIM, 2 * config.MODEL.DIM),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(2 * config.MODEL.DIM, config.DATA.NUM_CLASS)
                )

        elif self.mode == 'segmentation':
            self.upsampler = self.ssl_model.upsampler
            self.seg_head = nn.Conv2d(config.MODEL.UNET.DECODER_CHANNELS[-1], 
                                      config.DATA.NUM_CLASS,
                                      kernel_size=3, padding=3 // 2)
    
    def with_position_embedding(self, z):
        pos_embed = self.ch_pos.unsqueeze(0).expand(z.shape[0], -1, -1)  # (B, C, 256)
        z = z + pos_embed[:, :, None, :].expand(-1, -1, z.shape[2], -1)  # (B, C, N, 256)

        return z

    def forward(self, x):
        B, C, H, W = x.shape
        C_ = C // 3
        x = x.view(B * C_, 3, H, W)
        feature_list = self.encoder(x)

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
        src_flatten = torch.cat(src_flatten, 1).view(B, C_, -1, 256)  # (B, C_, N, 256)
        # src_flatten = self.with_position_embedding(src_flatten)  # (B, C_, N, 256)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # (4, 2)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  # (4)
        valid_ratios = torch.ones([B, spatial_shapes.shape[0], 2], device=src_flatten.device)  #(B, 4, 2)

        cr_experts, br_experts = self.decoder(src_flatten, spatial_shapes, level_start_index, valid_ratios)

        # classification
        if self.mode == 'classification':
            pred = self.mlp(cr_experts.flatten(1))  # (B, 3)

        # band regression
        elif self.mode == 'segmentation':
            H_ = W_ = int(br_experts.shape[1] ** 0.5)
            br_experts = br_experts.view(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()  # (B, 256, 16, 16)
            
            spatial_features = []
            for feature in feature_list[:-1]:
                c, h, w = feature.shape[-3:]
                feature = feature.view(B, -1, c, h, w)
                feature = torch.mean(feature, dim=1, keepdim=False)
                spatial_features.append(feature)
            spatial_features.append(br_experts)

            ft = self.upsampler(*spatial_features) # (B, 16, 512, 512)
            pred = self.seg_head(ft) # (B, 1, 512, 512)
            pred = nn.functional.interpolate(
                    pred, size=(H, W),
                    mode="bilinear", align_corners=False
                )

        return pred


def build_model(config):
    print("Init Model: {}\nBackbone: {}".format(config.MODEL.NAME, config.MODEL.BACKBONE))
    if config.MODEL.NAME == 'DeformableS3R':
        model = DeformableS3R(config)

    return model


def build_loss():
    BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)
    DiceLoss    = smp.losses.DiceLoss(mode='binary', log_loss=False)
    return {"BCELoss":BCELoss, "TverskyLoss":TverskyLoss, "DiceLoss":DiceLoss}
