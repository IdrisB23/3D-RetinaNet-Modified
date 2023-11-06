# https://github.com/hassony2/inflated_convnets_pytorch
# modified from ::=> credit: https://github.com/hassony2/inflated_convnets_pytorch/blob/master/src/inflate.py
import torch
import torch.nn as nn

from mmpretrain.models.utils import LayerNorm3d, LayerNorm2d
from mmpretrain.models.backbones import ConvNeXtBlock, ConvNeXtI3DBlock # changed source code
from collections import defaultdict


def inflate_conv(conv2d,
                 time_dim=4,
                 time_padding=1,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    groups = conv2d.groups
    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride,
        groups=groups)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim, dont_aggregate_time=True):
    """
    Args:
        time_dim: final time dimension of the features
    """
    out_features_3d = linear2d.out_features
    if dont_aggregate_time:
        in_features_3d = linear2d.in_features
    else:
        in_features_3d = linear2d.in_features * time_dim
    linear3d = nn.Linear(in_features_3d, out_features_3d)
    if dont_aggregate_time:
        weight3d = linear2d.weight.data
    else:
        weight3d = linear2d.weight.data.repeat(1, time_dim)
        weight3d = weight3d / time_dim

    linear3d.weight = nn.Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch3d._check_input_dim = batch2d._check_input_dim
    return batch3d


def inflate_layer_norm(layer2d: LayerNorm2d) -> LayerNorm3d:
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    layer3d = LayerNorm3d(layer2d.num_channels)
    return layer3d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    if isinstance(pool2d, nn.AdaptiveAvgPool2d):
        pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode)
        elif isinstance(pool2d, nn.AvgPool2d):
            pool3d = nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError('{} is not among known pooling classes'.format(type(pool2d)))

    return pool3d


def inflate_convnext_block(convnext_block: ConvNeXtBlock) -> ConvNeXtI3DBlock:
    raise NotImplementedError("inflate_convnext_block not implemented, yet")


def pretty_print_module_tree(module_tree: dict):
    descendents = module_tree["children"]
    if len(descendents) > 0:  
        for descendent in descendents:
            print(descendent["mod_name"])
            pretty_print_module_tree(descendent)


def inflate_all_model_submodules(model: nn.Module):
    print(f"Inflating model from 2D to 3D:")
    hierarchy_2_mod_names = defaultdict(list)
    mod_name_2_last_rsplit = dict()
    for mod_name, module in model.named_modules():
        rsplit_mod_name = mod_name.rsplit(".")
        # special case
        if rsplit_mod_name == [""]:
            hierarchy_index = 0
            mod_name_2_last_rsplit[mod_name] = None
        else:
            hierarchy_index = len(rsplit_mod_name)
            try:
                mod_name_2_last_rsplit[mod_name] = rsplit_mod_name[-2:]
            except Exception:
                mod_name_2_last_rsplit[mod_name] = rsplit_mod_name[-1:]
            
        hierarchy_2_mod_names[hierarchy_index].append(mod_name)
    
    
    mod_name_hierarchy_tree = dict(children=[])
    mod_name_2_tree_node = dict()
    # build tree
    for hier_idx, mod_names in sorted(hierarchy_2_mod_names.items()):
        for mod_name in mod_names:
            if mod_name == "":
                mod_name = "root"
            tree_node = dict(mod_name=mod_name, children=[])
            if hier_idx == 0:
                parent_tree_node = mod_name_hierarchy_tree
            elif hier_idx == 1:
                parent_tree_node = mod_name_2_tree_node["root"]
            else:
                # find parent
                parent_idx = hier_idx - 1
                potential_parent_names = hierarchy_2_mod_names[parent_idx]
                # filter out parents by checking whether their last rsplits
                nb_parents = 0
                for potential_parent_n in potential_parent_names:
                    parent_last_rsplit = mod_name_2_last_rsplit[potential_parent_n]   
                    current_module_last_rsplit = mod_name_2_last_rsplit[mod_name]
                    if current_module_last_rsplit[-2] == parent_last_rsplit[-1]:
                        parent_tree_node = mod_name_2_tree_node[potential_parent_n]
                        # it should have max 1 parent
                        nb_parents += 1
            
            parent_tree_node["children"].append(tree_node)
            #tree_node["parent"] = parent_tree_node
            mod_name_2_tree_node[mod_name] = tree_node


    #pretty_print_module_tree(mod_name_hierarchy_tree)
    pass