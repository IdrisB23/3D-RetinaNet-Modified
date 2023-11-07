from models.backbones import swin_tiny_config

from models.resnetFPN import conv1x1, conv3x3, _upsample

import torch.nn as nn

import torchvision.transforms as transforms
PIL2tensor = transforms.Compose([transforms.ToTensor()])
tensor2PIL = transforms.Compose([transforms.ToPILImage()])


class FPN(nn.Module):
    def __init__(self, backbone_constructor, constructor_kwargs):
        super(FPN, self).__init__()
        self.backbone = backbone_constructor(**constructor_kwargs)
        
        # the nb of channels in the output feature maps depend on the size of the model, for now hard-coded
        self.lateral_layer1 = conv1x1(768, 256) # conv1x1(2304, 256)
        self.lateral_layer2 = conv1x1(384, 256)
        self.lateral_layer3 = conv1x1(192, 256)

        self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P5
        self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P4
        self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1)  # P3

        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))


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
        # for now remove, because in the orig paper they removed last convolutions from backbones and replaced them?
        # p6 = self.conv6(c5)
        # p7 = self.conv7(F.relu(p6))
        # features = [p3, p4, p5, p6, p7]
        # ego_feat = self.avg_pool(p7)
        features = [p3, p4, p5]
        ego_feat = self.avg_pool(p5)
        return features, ego_feat


    def load_my_state_dict(self, state_dict):
        pass