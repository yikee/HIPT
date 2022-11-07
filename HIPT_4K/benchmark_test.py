import h5py
from openslide import open_slide
from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from hipt_heatmap_utils import *
import glob
import json

wsi_fpath = 'wsi_3.svs'
h5_fpath = "wsi_3.h5"

wsi = open_slide(wsi_fpath)
h5 = h5py.File(h5_fpath, 'r')
patch_coords = h5['coords']
num_4k_patches = len(patch_coords)
patch_level = patch_coords.attrs['patch_level']
patch_size = patch_coords.attrs['patch_size']
for i in range(num_4k_patches):
    img = wsi.read_region(patch_coords[i], patch_level, (patch_size, patch_size)).convert('RGB')
    img.save('4k_imgs/wsi_' + str(i) + '.png')

pretrained_weights256 = 'HIPT_4K/Checkpoints/vit256_small_dino.pth'
pretrained_weights4k = 'HIPT_4K/Checkpoints/vit4k_xs_dino.pth'
device256 = torch.device('gpu')
device4k = torch.device('gpu')

### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

### ViT_256 + ViT_4K loaded into HIPT_4K API
model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
model.eval()

res = {}
for filename in glob.glob('4k_imgs/*.png'):
    region = Image.open(filename)
    x = eval_transforms()(region).unsqueeze(dim=0)
    embeddings = model.forward(x)
    res[str(filename)] = (x, embeddings)
    print('Input Shape:', x.shape)
    print('Output Shape:', res.shape)

with open("embeddings.json", "w") as outfile:
    json.dump(res, outfile)