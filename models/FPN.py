from models.backbones import swin_tiny_config

from models.resnetFPN import conv1x1, conv3x3, _upsample

import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
PIL2tensor = transforms.Compose([transforms.ToTensor()])
tensor2PIL = transforms.Compose([transforms.ToPILImage()])


class FPN(nn.Module):
    follow_construct_2_block_return_channels = dict(
        swin_t={"-4": 96, "-3": 192, "-2": 384, "-1": 768}, 
        convnext_t={"-4": 96, "-3": 192, "-2": 384, "-1": 768}
        )
    
    def __init__(self, backbone_constructor, constructor_kwargs, follow_construct="convnext_t"):
        super(FPN, self).__init__()

        self.backbone = backbone_constructor(**constructor_kwargs)
        block_return_channels = FPN.follow_construct_2_block_return_channels[follow_construct] # get_return_channels_of_each_FE_stage
        
        self.conv6 = conv3x3(block_return_channels["-1"], 256, stride=2, padding=1)  # P6
        self.conv7 = conv3x3(256, 256, stride=2, padding=1)  # P7

        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.lateral_layer1 = conv1x1(block_return_channels["-1"], 256)
        self.lateral_layer2 = conv1x1(block_return_channels["-2"], 256)
        self.lateral_layer3 = conv1x1(block_return_channels["-3"], 256)
        
        self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P4
        self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P4
        self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1)  # P3


    def init_weights(self, init_hook=None, hook_kwargs=None):
        if init_hook is None:
            self.backbone.init_weights()
        else:
            if hook_kwargs is None:
                init_hook(self)
            else:
                init_hook(self, hook_kwargs)


    def forward(self, x):
        out_feature_maps = self.backbone(x) # (B, C, W, H)
        # because ViT works on 2D (for now), expand T dim
        lastFM1 = out_feature_maps[-1]
        lastFM2 = out_feature_maps[-2]
        lastFM3 = out_feature_maps[-3]
        c3 = lastFM3
        c4 = lastFM2
        c5 = lastFM1
        p5 = self.lateral_layer1(c5)
        p5_upsampled = _upsample(p5, c4)
        p5 = self.corr_layer1(p5)
        p4 = self.lateral_layer2(c4)
        p4 = p5_upsampled + p4
        p4_upsampled = _upsample(p4, c3)
        p4 = self.corr_layer2(p4)
        p3 = self.lateral_layer3(c3)
        p3 = p4_upsampled + p3
        p3 = self.corr_layer3(p3)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        features = [p3, p4, p5, p6, p7]
        ego_feat = self.avg_pool(p7)
        # for now remove, because in the orig paper they removed last convolutions from backbones and replaced them? (old)
        # replicate code for slowfast as it utilises FMs from backbone (new): added p6 and p7
        return features, ego_feat


    def load_my_state_dict(self, state_dict):
        pass