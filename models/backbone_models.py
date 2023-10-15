from models.FPN import FPN
from models.resnetFPN import resnetfpn
from models.backbones import swin_tiny_config, swin_small_config
import torch
from modules.utils import get_logger

logger = get_logger(__name__)

def backbone_models(args):
    base_arch, MODEL_TYPE = args.ARCH, args.MODEL_TYPE
    print(base_arch)

    assert base_arch.startswith("resnet") or base_arch.startswith("swin") or base_arch.startswith("convnext")

    modelperms = {'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], "swint": [2, 2, 6, 2], "swins": [2, 2, 18, 2]}
    model_configs = {"swint": swin_tiny_config, "swins": swin_small_config}
    model_3d_layers = {'resnet50': [[0, 1, 2], [0, 2], [0, 2, 4], [0, 1]], 
                       'resnet101': [[0, 1, 2], [0, 2], [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], [0, 1]]}
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
        config = model_configs[base_arch.lower()]
        logger.info(config)
        model = FPN(config)
    else:
        raise RuntimeError("Define the argument --ARCH correclty:: " + args.ARCH)

    if args.MODE == 'train':
        if not base_arch.lower().startswith("resnet"):
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
