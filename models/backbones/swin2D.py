from mmdet.models.backbones import SwinTransformer
from torch import Tensor
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt

PIL2tensor = transforms.Compose([transforms.ToTensor()])
tensor2PIL = transforms.Compose([transforms.ToPILImage()])


def swin_transformer(**kwargs):
    return SwinTransformer(**kwargs)


def save_each_feature_map_in_dir(features: Tensor, dir: Path= Path("feature_maps")):
    print(features.shape)
    for feat_nb in range(features.shape[0]):
        image_file_name = f"{features.shape[0]}feature_nb_{feat_nb}.png"
        img_tb_saved = tensor2PIL(features[feat_nb])
        img_tb_saved.save((dir / image_file_name).as_posix())

def plot_images_in_grid_and_save_to_dir(image: Tensor, dir: Path= Path("feature_maps")):
    n_channels = image.shape[0]
    square = int(math.ceil(math.sqrt(n_channels)))
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            if ix-1 >= n_channels:
                break
            plt.imshow(image[ix-1, :, :], cmap='gray')
            ix += 1
    # show the figure
    plt.savefig(dir / f'{n_channels}_feature_maps.png')

def main():
    backbons_DIR = Path("models") / "backbones"
    swin_small_pretrained_p_ = backbons_DIR / "pretrained_models" / "swin_small_patch4_window7_224.pth"
    example_image_p_ = backbons_DIR / "dogs_1280p_0.jpg"

    swin_transformer_init_config = dict(_delete_=True,
            type='SwinTransformer',
            embed_dims=96,
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
            convert_weights=True,
            init_cfg=dict(type='Pretrained', checkpoint=swin_small_pretrained_p_))

    model = swin_transformer(
        #depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), strides=(4, 2, 2, 2, 2), out_indices=(0, 1, 2, 3), 
                            init_cfg=swin_transformer_init_config)

    expl_img = Image.open(example_image_p_)
    expl_img_tensor = PIL2tensor(expl_img).unsqueeze(0)

    model_output = model(expl_img_tensor)
    for out in model_output:
        out_detached = out[0].detach().cpu()
        #plot_images_in_grid_and_save_to_dir(out_detached)
        print(out_detached.shape)
        #save_each_feature_map_in_dir(out_detached)

if __name__ == "__main__":
    main()