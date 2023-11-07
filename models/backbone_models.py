from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent.parent / "mmpretrain"))
sys.path.insert(1, str(Path(__file__).parent.parent / "mmpretrain_custom"))

import torch

from modules.utils import get_logger

from models.FPN import FPN
from models.resnetFPN import resnetfpn

from models.backbones import swin_tiny_config, swin_small_config
from models.backbones.swin_transformer.swin3D import SwinTransformer3D

from mmpretrain.models.backbones import ConvNeXt
from mmpretrain_custom.models.backbones.convnext import ConvNeXtI3D
from models.backbones.convnext.convnext import return_model_with_inflated_weights
from models.backbones import convnext_tiny_constructor_kwargs
from models.backbones import convnext_small_constructor_kwargs
from models.backbones import convnext_base_constructor_kwargs


logger = get_logger(__name__)

def backbone_models(args):
    base_arch, MODEL_TYPE = args.ARCH, args.MODEL_TYPE
    print(base_arch)

    assert base_arch.startswith("resnet") \
        or base_arch.startswith("swin") \
            or base_arch.startswith("internimage") \
                or base_arch.startswith("convnext")

    modelperms = {
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        "swint": [2, 2, 6, 2],
        "swins": [2, 2, 18, 2], 
        "convnext_t": None, # placeholder
        }
    
    model_3d_layers = {'resnet50': [[0, 1, 2], [0, 2], [0, 2, 4], [0, 1]], 
                       'resnet101': [[0, 1, 2], [0, 2], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [0, 1]]}
    swin_constructor_kwargs_dict = {"swint": swin_tiny_config, "swins": swin_small_config}
    convnext_constructor_kwargs_dict = {"convnext_t": convnext_tiny_constructor_kwargs,
                                        "convnext_s": convnext_small_constructor_kwargs,
                                        "convnext_b": convnext_base_constructor_kwargs,
                                        }

    assert base_arch in modelperms, 'Arch shoudl from::>' + \
        ','.join([m for m in modelperms])

    if args.MODEL_TYPE.endswith('-NL'):
        args.non_local_inds = [[], [1, 3], [1, 3, 5], []]
    else:
        args.non_local_inds = [[], [], [], []]

    if base_arch.lower().startswith("resnet"):
        args.model_perms = modelperms[base_arch]
        args.model_3d_layers = model_3d_layers[base_arch]
        model = resnetfpn(args)
    elif base_arch.lower().lower().startswith("swin"):
        swin_kwargs = swin_constructor_kwargs_dict[base_arch.lower()]
        logger.info(swin_kwargs)
        model = FPN(SwinTransformer3D, swin_kwargs)
    elif base_arch.lower().startswith("convnext"):
        convnext_kwargs = convnext_constructor_kwargs_dict[base_arch.lower()]
        logger.info(convnext_kwargs)
        if args.MODE == 'train':
            logger.info("init with ConvNeXt!!!")
            model = FPN(ConvNeXt, convnext_kwargs) # will be inflated later
        else:
            model = FPN(ConvNeXtI3D, convnext_kwargs)
    else:
        raise RuntimeError("Define the argument --ARCH correclty:: " + args.ARCH)

    if args.MODE == 'train':
        logger.info("training!!!")
        if not base_arch.lower().startswith("resnet"):
            if base_arch.lower().startswith("convnext"):
                def init_weights_func(ref_to_FPN, kwargs):
                    dummy_inflated_convnext = ConvNeXtI3D(**convnext_kwargs)
                    FPN_convnext_2D_backbone = ref_to_FPN.backbone
                    logger = kwargs["logger"]
                    logger.info("FPN_convnext_2D_backbone.__class__", FPN_convnext_2D_backbone.__class__)
                    # first init weights for 2D model
                    FPN_convnext_2D_backbone.init_weights()
                    # inflate the models and change 2D parameters/layers to their appropriate 3D counterparts
                    inflated_backbone = return_model_with_inflated_weights(
                        FPN_convnext_2D_backbone, 
                        dummy_inflated_convnext, 
                        time_dim=args.SEQ_LEN
                        )
                    ref_to_FPN.backbone = inflated_backbone
                model.init_weights(init_weights_func, dict(logger=logger))
            model.init_weights()
        else:
            if MODEL_TYPE.startswith('RCN'):
                model.identity_state_dict()
            if MODEL_TYPE.startswith('RCGRU') or MODEL_TYPE.startswith('RCLSTM'):
                model.recurrent_conv_zero_state()
            if not MODEL_TYPE.startswith('SlowFast'):
                load_dict = torch.load(args.MODEL_PATH)
                model.load_my_state_dict(load_dict)

    return model
