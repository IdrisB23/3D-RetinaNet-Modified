import torch, pdb
import torch.optim as optim
# from .madamw import Adam as AdamM
# from .adamw import Adam as AdamW
from models.backbones import freeze_layers_convnext
from modules import utils
logger = utils.get_logger(__name__)

from torch.optim.lr_scheduler import MultiStepLR

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, MILESTONES, GAMMAS, last_epoch=-1):
        self.MILESTONES = MILESTONES
        self.GAMMAS = GAMMAS
        assert len(GAMMAS) == len(MILESTONES), 'MILESTONES and GAMMAS should be of same length GAMMAS are of len ' + (len(GAMMAS)) + ' and MILESTONES '+ str(len(MILESTONES))
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.MILESTONES:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            index = self.MILESTONES.index(self.last_epoch)
            return [group['lr'] * self.GAMMAS[index] for group in self.optimizer.param_groups]
    
    #def print_lr(self):
    #   print([[group['name'], group['lr']] for group in self.optimizer.param_groups])

def freeze_layers_resnet(args, net):
    freeze_layers = ['backbone_net.layer'+str(n) for n in range(1, args.FREEZE_UPTO+1)]
    solver_print_str = '\n\nSolver configs are as follow \n\n\n'
    params = []
    for key, value in net.named_parameters():
        if args.FREEZE_UPTO>0 and (key.find('backbone.conv1')>-1 or key.find('backbone.bn1')>-1): # Freeze first conv layer and bn layer in resnet
            value.requires_grad = False
            continue
        if key.find('backbone')>-1: 
            for layer_id in freeze_layers:
                if key.find(layer_id)>-1:
                    value.requires_grad = False    
                    continue
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


def get_optim(args, net):
    if "resnet" in args.ARCH:
        params, solver_print_str = freeze_layers_resnet(args, net)
    elif "convnext" in args.ARCH:
        params, solver_print_str = freeze_layers_convnext(args, net)

    if args.OPTIM == 'SGD':
        optimizer = optim.SGD(params)
    elif args.OPTIM == 'ADAM':
        optimizer = optim.Adam(params)
    else:
        raise NotImplementedError('Define optimiser type')

    solver_print_str += 'optimizer is '+ args.OPTIM + '\nDone solver configs\n\n'
    #scheduler = WarmupMultiStepLR(optimizer, args.MILESTONES, args.GAMMAS)
    scheduler = MultiStepLR(optimizer, args.MILESTONES, args.GAMMA)

    return optimizer, scheduler, solver_print_str
