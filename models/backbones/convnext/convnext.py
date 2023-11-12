import torch
import torch.nn as nn

from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(1, str(Path(__file__).parent.parent.parent.parent / "mmpretrain"))
sys.path.insert(1, str(Path(__file__).parent.parent.parent.parent / "mmpretrain_custom"))

import models.backbones.inflate_2D_weights as inflate

import mmpretrain
from mmpretrain.models.backbones import ConvNeXt
from mmpretrain_custom.models.backbones.convnext import ConvNeXtI3D # changed source code


from modules import utils
logger = utils.get_logger(__name__)


PRTRAIND_DIR = Path("pre-trained")
IN_DIR = PRTRAIND_DIR / "imagenet-pt"
IN_1K_DIR = IN_DIR / "1k"
IN_21K_DIR = IN_DIR / "21k"
convnext_tiny_in1k_ema_weights_P_ = IN_1K_DIR / "convnext-tiny_32xb128_ema_in1k_20221207-998cf3e9.pth"

convnext_tiny_constructor_kwargs = dict(
    arch= "tiny",
    out_indices=[-4, -3, -2, -1], # last 4 FMs
    gap_before_final_norm=False,
    frozen_stages=2,
    init_cfg= dict(
       type= "Pretrained",
       checkpoint= convnext_tiny_in1k_ema_weights_P_.as_posix(),
       prefix= "backbone."
    )
)
convnext_small_constructor_kwargs = None # todo
convnext_base_constructor_kwargs = None # todo


def calc_kernel_size_padding_to_preserve_output_time_dim(time_dim: int) -> dict:
    # is even
    if time_dim % 2 == 0:
        time_dim = time_dim-1
    # is odd
    else:
        pass
    return dict(time_dim=time_dim, time_padding=time_dim // 2)


def return_model_with_inflated_weights\
(convnext_model_2d: nn.Module, dummy_convnexti3D_same_size: nn.Module, time_dim: int = 1) -> nn.Module:
    new_model = ConvNeXtI3D(**convnext_tiny_constructor_kwargs)
    convnext_named_children = [(name, mod) for name, mod in convnext_model_2d.named_children()]
    for module_name, module in convnext_named_children:
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
                conv3d = inflate.inflate_conv(conv2d, **calc_kernel_size_padding_to_preserve_output_time_dim(time_dim))
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
                            new_module = inflate.inflate_conv(module, **calc_kernel_size_padding_to_preserve_output_time_dim(time_dim))
                        elif module.__class__ == mmpretrain.models.utils.norm.LayerNorm2d:
                            new_module = inflate.inflate_layer_norm(module)
                        else:
                            print(f"Skipping submodule of class {module.__class__}")
                            continue
                        setattr(dummy_convnext_block, module_name, new_module)
                    convnext_blocks.append(dummy_convnext_block)
                convnext_blocks_sequential = nn.Sequential(*convnext_blocks)
                new_stages.append(convnext_blocks_sequential)
            new_stages_module_list = nn.ModuleList(new_stages)
            setattr(new_model, "stages", new_stages_module_list)
        else:
            print(f"Module is neither of name 'downsample' nor 'stages', but rather {module_name}")
    return new_model


def _check_layers_are_frozen(convnext):
    logger.info("freeze_layers_convnext | net.named_children:")
    # 2 cases for convnextI3D: downsample_layers | stages
    downsample_layers = convnext.downsample_layers # nn.Sequential
    for i, seq in enumerate(downsample_layers):
        print(i)
        if i == 0:
            ln = seq[1]
            conv = seq[0]
        else:
            ln = seq[0]
            conv = seq[1]
        logger.info("ln.bias | " + str(ln.bias.requires_grad))
        logger.info("ln.weight | " + str(ln.weight.requires_grad))
        logger.info("conv.weight | " + str(conv.weight.requires_grad))
        logger.info("conv.bias | " + str(conv.bias.requires_grad))
    stages = convnext.stages # nn.ModuleList
    for i, seq in enumerate(stages):
        pass # todo
    pass


def freeze_layers_convnext(args, net):
    # __init__ of ConvNeXtI3D already calls _freeze_layers but somehow the parameters don't 
    # seem to be frozen. Suspect this might be due to cloning and inflating them from ConvNeXt
    # into ConvNeXtI3D
    frozen_layers = args.FREEZE_UPTO
    FPN = net.backbone # FPN
    convnext = FPN.backbone # convnext
    setattr(convnext, "frozen_stages", frozen_layers)
    convnext._freeze_stages()
    _check_layers_are_frozen(convnext) # comment/uncomment
    solver_print_str = '\n\nSolver configs are as follow \n\n\n'
    params = []
    logger.info("convnext.named_parameters():")
    for key, value in convnext.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.LR
        wd = args.WEIGHT_DECAY
        if args.OPTIM == 'ADAM':
            wd = 0.0
        if "bias" in key:
            lr = lr*2.0
        if args.OPTIM == 'SGD':
            params += [{"params": [value], "name":key, "lr": lr, "weight_decay":wd, "momentum":args.MOMENTUM}]
        else:
            params += [{"params": [value], "name":key, "lr": lr, "weight_decay":wd}]
        print_l = key +' is trained at the rate of ' + str(lr)
        solver_print_str += print_l + '\n'
    return params, solver_print_str


def inflate_convnext_and_init_weights_func(ref_to_FPN, kwargs):
    args = kwargs["args"]
    logger = kwargs["logger"]
    convnext_kwargs = kwargs["convnext_kwargs"]
    dummy_inflated_convnext = ConvNeXtI3D(**convnext_kwargs)
    FPN_convnext_2D_backbone = ref_to_FPN.backbone
    # first init weights for 2D model
    logger.info("init_weights for FPN_convnext_2D_backbone")
    FPN_convnext_2D_backbone.init_weights()
    # inflate the models and change 2D parameters/layers to their appropriate 3D counterparts
    logger.info("Inflating ConvNeXt backbone")
    inflated_backbone = return_model_with_inflated_weights(
        FPN_convnext_2D_backbone, 
        dummy_inflated_convnext, 
        time_dim=args.SEQ_LEN
        )
    # logger.info("Comparing the models:")
    # inflated_named_children = dict(inflated_backbone.named_children())
    # for name, module in FPN_convnext_2D_backbone.named_children():
    #     if name.lower().startswith("downsample"):
    #         logger.info("Comparing submodule-downsample_layers")
    #         downsample_layers = module # of class nn.ModuleList
    #         dowsample_inflated = inflated_named_children["downsample_layers"]
    #         # all submodules are of class nn.Sequential
    #         # and have the same structure, except for the first one (stem)
    #         is_stem = True
    #         for k, seq_module in enumerate(downsample_layers):
    #             # handle stem first
    #             seq_module_inflated = dowsample_inflated[k]
    #             if is_stem:
    #                 conv2d = seq_module[0]
    #                 conv3d = seq_module_inflated[0]
    #                 is_stem = False
    #             else:
    #                 conv2d = seq_module[1]
    #                 conv3d = seq_module_inflated[1]
    #             # compare weights and biases of convs
    #             weight_2d = conv2d.weight.data
    #             bias_2d = conv2d.bias
    #             weight_3d = conv3d.weight.data
    #             bias_3d = conv3d.bias
    #             logger.info("bias_2d==bias_3d: " + str(bias_2d==bias_3d))
    #     elif name.lower().startswith("stages"):
    #         logger.info("Comparing submodule-stages")
    #         stages = module # of class nn.ModuleList
    #         stages_inflated = inflated_named_children["stages"]
    #         for i, seq_module in enumerate(stages):
    #             convnext_blocks = []
    #             dummy_seq_convnext_block = dummy_stages[i]
    #             for j, convnext_block in enumerate(seq_module):
    #                 dummy_convnext_block = dummy_seq_convnext_block[j]
    #                 for module_name, module in convnext_block.named_children():
    #                     if module.__class__ == nn.modules.linear.Linear:
    #                         new_module = inflate.inflate_linear(module, time_dim)
    #                     elif module.__class__ == nn.modules.conv.Conv2d:
    #                         new_module = inflate.inflate_conv(module, **calc_kernel_size_padding_to_preserve_output_time_dim(time_dim))
    #                     elif module.__class__ == mmpretrain.models.utils.norm.LayerNorm2d:
    #                         new_module = inflate.inflate_layer_norm(module)
    #                     else:
    #                         print(f"Skipping submodule of class {module.__class__}")
    #                         continue
    #                     setattr(dummy_convnext_block, module_name, new_module)
    #                 convnext_blocks.append(dummy_convnext_block)
    #             convnext_blocks_sequential = nn.Sequential(*convnext_blocks)
    #             new_stages.append(convnext_blocks_sequential)
    #         new_stages_module_list = nn.ModuleList(new_stages)
    #         setattr(new_model, "stages", new_stages_module_list)
    #     else:
    #         print(f"Cannot compare as module is neither of name 'downsample' nor 'stages', but rather {name}")
    ref_to_FPN.backbone = inflated_backbone



def main():
    # doesn't work with ConvNextI3D, undeterministic init of weights
    model = ConvNeXt(**convnext_tiny_constructor_kwargs)
    dummy_model_convnextI3D = ConvNeXtI3D(**convnext_tiny_constructor_kwargs)
    model.init_weights()
    new_model = return_model_with_inflated_weights(model, dummy_model_convnextI3D, time_dim=4)
    print(new_model)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    input_ = torch.randn((5, 3, 4, 224, 224))
    output_ = new_model(input_)
    for out in output_:
        print('out:', out.shape)

if __name__ == "__main__":
    main()