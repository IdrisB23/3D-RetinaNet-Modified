import torch.nn as nn

from .slowfast import *
from utils import load_pretrain

from pathlib import Path

pretrained_DIR = Path("pre-trained")
swin_t_ImageNet1K_2D_Weights_P_ = pretrained_DIR / "imagenet-pt" / "swin_tiny_patch4_window7_224.pth"
swin_s_ImageNet1K_2D_Weights_P_ = pretrained_DIR / "imagenet-pt" / "swin_small_patch4_window7_224.pth"

swin_tiny_config = {"depths": [2, 2, 6, 2], "embed_dim": 96, "pretrained": swin_t_ImageNet1K_2D_Weights_P_.as_posix()}
swin_small_config = {"depths": [2, 2, 18, 2], "embed_dim": 96, "pretrained": swin_s_ImageNet1K_2D_Weights_P_.as_posix()}
swin_big_config = {"depths": [2, 2, 18, 2], "embed_dim": 128}
swin_large_config = {"depths": [2, 2, 18, 2], "embed_dim": 192}


def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])


class AVA_backbone(nn.Module):
    def __init__(self, config):
        super(AVA_backbone, self).__init__()
        
        self.config = config
        self.module = model_entry(config.model.backbone)
        print(config.get('pretrain', None))
        if config.get('pretrain', None) is not None:
            load_pretrain(config.pretrain, self.module)
                
        if not config.get('learnable', True):
            self.module.requires_grad_(False)

    # data: clips
    # returns: features
    def forward(self, data):
        # inputs = data['clips']
        inputs = data
        inputs = inputs.cuda()
        features = self.module(inputs)
        return features
