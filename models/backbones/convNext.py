import mmpretrain
from mmpretrain.models.backbones import ConvNeXt, ConvNeXtI3D #package installed from source
import torch
import torch.nn as nn
import inflate
from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
from models.backbones import convnext_tiny_constructor_kwargs

def return_model_with_inflated_weights(convnext_model_2d: nn.Module, dummy_convnexti3D_same_size: nn.Module, time_dim: int = 1) -> nn.Module:
    new_model = ConvNeXtI3D(**convnext_tiny_constructor_kwargs)
    convnext_named_children = [(name, mod) for name, mod in convnext_model_2d.named_children()]
    for module_name, module in convnext_named_children:
        print(module_name.lower())
        if module_name.lower().startswith("downsample"):
            downsample_layers = module # of class nn.ModuleList
            # all submodules are of class nn.Sequential
            # and have the same structure, except for the first one (stem)
            is_stem = True
            new_downsample_layers = []
            for seq_module in downsample_layers:
                # handle stem first
                if is_stem:
                    ln2 = seq_module[1]
                    conv2d = seq_module[0]
                else:
                    ln2 = seq_module[0]
                    conv2d = seq_module[1]
                # inflate as per I3D
                ln3 = inflate.inflate_layer_norm(ln2)
                conv3d = inflate.inflate_conv(conv2d, time_dim)
                if is_stem:
                    is_stem = False
                    new_seq_module = nn.Sequential(*[conv3d, ln3])
                else:
                    new_seq_module = nn.Sequential(*[ln3, conv3d])
                new_downsample_layers.append(new_seq_module)
            new_downsample_layers = nn.ModuleList(new_downsample_layers)
            setattr(new_model, "downsample_layers", new_downsample_layers)
        elif module_name.lower().startswith("stages"):
            stages = module # of class nn.ModuleList
            dummy_stages = getattr(dummy_convnexti3D_same_size, "stages")
            new_stages = []
            for i, seq_module in enumerate(stages):
                convnext_blocks = []
                dummy_seq_convnext_block = dummy_stages[i]
                for j, convnext_block in enumerate(seq_module):
                    dummy_convnext_block = dummy_seq_convnext_block[j]
                    for module_name, module in convnext_block.named_children():
                        if module.__class__ == nn.modules.activation.GELU:
                            new_module = module
                        elif module.__class__ == nn.modules.linear.Identity:
                            new_module = module
                        elif module.__class__ == nn.modules.linear.Linear:
                            new_module = inflate.inflate_linear(module, time_dim)
                        elif module.__class__ == nn.modules.conv.Conv2d:
                            new_module = inflate.inflate_conv(module, time_dim)
                        elif module.__class__ == mmpretrain.models.utils.norm.LayerNorm2d:
                            new_module = inflate.inflate_layer_norm(module)
                        else:
                            print("Somethings wrong")
                            continue
                        setattr(dummy_convnext_block, module_name, new_module)
                    convnext_blocks.append(dummy_convnext_block)
                convnext_blocks_sequential = nn.Sequential(*convnext_blocks)
                new_stages.append(convnext_blocks_sequential)
            new_stages_module_list = nn.ModuleList(new_stages)
            setattr(new_model, "stages", new_stages_module_list)
        else:
            print("Somethings wrong 2")
    return new_model

def main():
    # doesn't work with ConvNextI3D, undeterministic init of weights
    model = ConvNeXt(**convnext_tiny_constructor_kwargs)
    dummy_model_convnextI3D = ConvNeXtI3D(**convnext_tiny_constructor_kwargs)
    model.init_weights()
    new_model = return_model_with_inflated_weights(model, dummy_model_convnextI3D)
    print(new_model)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    input_ = torch.randn((5, 3, 8, 224, 224))
    output_ = new_model(input_)
    for out in output_:
        print('out:', out.shape)

if __name__ == "__main__":
    main()