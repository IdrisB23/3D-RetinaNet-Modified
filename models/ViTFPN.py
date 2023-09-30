from models.backbones.swin2D import swin_transformer
from models.backbones.swin3D import SwinTransformer3D, swin_tiny_config
from models.resnetFPN import conv1x1, conv3x3, _upsample, _upsample_time

import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


from PIL import Image
import torchvision.transforms as transforms
PIL2tensor = transforms.Compose([transforms.ToTensor()])
tensor2PIL = transforms.Compose([transforms.ToPILImage()])

backbones_DIR = Path("models") / "backbones"
swin_small_pretrained_P_ = backbones_DIR / "pretrained_models" / "swin_small_patch4_window7_224.pth"
example_img_P_ = backbones_DIR / "dogs_1280p_0.jpg"
swin_transformer2d_config = dict(_delete_=True,
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            pretrained_model_path=swin_small_pretrained_P_)


class ViTFPN(nn.Module):
    def __init__(self, embed_dims, depths, num_heads, window_size, 
                 mlp_ratio, out_indices, pretrained_model_path: Path=None, **kwargs):
        super(ViTFPN, self).__init__()
        # the specified pretrained_model should be compartible with the Swin Transformer size / depth
        # config = dict(type='SwinTransformer',
        #     embed_dims=embed_dims,
        #     depths=depths,
        #     num_heads=num_heads,
        #     window_size=window_size,
        #     mlp_ratio=mlp_ratio,
        #     out_indices=out_indices,
        #     convert_weights=False if pretrained_model_path is None else True,
        #     init_cfg=dict(
        #         type='Pretrained',
        #         checkpoint=pretrained_model_path
        #         ) if not pretrained_model_path is None else None,
        #     **kwargs)   
        # self.backbone = swin_transformer(init_cfg=config)
        # self.nb_out_feature_maps = len(config["out_indices"])
        # assert self.nb_out_feature_maps > 2, "Please make sure the ViT is deep enough to produce at least 3 Feature Maps"
        self.backbone = SwinTransformer3D(**swin_tiny_config)
        
        # the nb of channels in the output feature maps depend on the size of the ViT, for now hard-coded
        self.lateral_layer1 = conv1x1(768, 256) # conv1x1(2304, 256)
        self.lateral_layer2 = conv1x1(384, 256)
        self.lateral_layer3 = conv1x1(192, 256)

        self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P5
        self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P4
        self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1)  # P3

        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))


    def init_weights(self):
        self.backbone.init_weights()


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

def vitfpn(args):
    return ViTFPN(**swin_transformer2d_config)


def main():
    example_img = Image.open(example_img_P_.as_posix())

    example_img_tensor = PIL2tensor(example_img).unsqueeze(0) # (B, C, W, H)
    import torch
    example_img_tensor = torch.randn((3, 3, 180, 320))
    img_expanded_to_vid = example_img_tensor.unsqueeze(2).repeat(1, 1, 8, 1, 1) # (B, C, T=8, W, H) 
    print(img_expanded_to_vid.shape)
    model = ViTFPN(**swin_transformer2d_config)
    features, ego_feat = model(img_expanded_to_vid)
    for i, feature in enumerate(features):
        print(f"feature[{i}]:", feature.shape)
    print("ego_feat:", ego_feat.shape)