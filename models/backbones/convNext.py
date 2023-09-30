from mmpretrain.models.backbones import ConvNeXtI3D #package installed from source
import torch

def main():
    model = ConvNeXtI3D(arch="small", out_indices=[-1, -2, -3])
    input_ = torch.randn(1, 3, 8, 224, 224)
    output_ = model(input_)
    for out in output_:
        print('out:', out.shape)