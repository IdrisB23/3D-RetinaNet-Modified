from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent.parent.parent.parent))

from models.backbones.inflate_2D_weights.inflate_modules import inflate_conv, inflate_linear, inflate_batch_norm, inflate_layer_norm, inflate_pool