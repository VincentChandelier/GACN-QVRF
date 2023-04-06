import os

from compressai.zoo import models
from compressai.models import (
    Cheng2020Anchor,
    Cheng2020Attention,
    FactorizedPrior,
    JointAutoregressiveHierarchicalPriors,
    MeanScaleHyperprior,
    ScaleHyperprior,
)
from Network import *
from pathlib import Path
from typing import Dict, Tuple
from torch import Tensor

def getmodel(models, quality):
    if models == 'Cheng2020Anchor':
        return Cheng2020Anchor(128,)
    elif models == 'Cheng2020Attention':
        return Cheng2020Attention(128, )
    elif models == 'Factor':
        return FactorizedPrior(128, 192)
    elif models == 'Hyper':
        return ScaleHyperprior(128, 192)
    elif models == 'Joint':
        return JointAutoregressiveHierarchicalPriors(192, 192)
    # if models == 'Cheng2020Anchor':
    #     if quality < 4:
    #         return Cheng2020Anchor(128,)
    #     else:
    #         return Cheng2020Anchor(192, )
    # elif models == 'Cheng2020Attention':
    #     if quality < 4:
    #         return Cheng2020Attention(128, )
    #     else:
    #         return Cheng2020Attention(192, )
    # elif models == 'Factor':
    #     if quality < 4:
    #         return FactorizedPrior(128, 192)
    #     else:
    #         return FactorizedPrior(192, 320)
    # elif models == 'Hyper':
    #     if quality < 4:
    #         return ScaleHyperprior(128, 192)
    #     else:
    #         return ScaleHyperprior(192, 320)
    # elif models == 'Joint':
    #     if quality < 4:
    #         return JointAutoregressiveHierarchicalPriors(192, 192)
    #     else:
    #         return JointAutoregressiveHierarchicalPriors(192, 320)
    elif models == 'Manual':
        if quality < 4:
            return testNetwork(192, 192)
        else:
            return testNetwork(192, 320)
    else:
        print("no suitable model")
        exit(1)

def DelfileList(path, filestarts='checkpoint_last'):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(filestarts):
                os.remove(os.path.join(root, file))

def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    return state_dict

from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F

class LensletRandomCrop(RandomCrop):

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        view_size = 48
        i = torch.randint(0, (h - th)//view_size + 1, size=(1,)).item()*view_size
        j = torch.randint(0, (w - tw) //view_size+ 1, size=(1,)).item()*view_size
        return i, j, th, tw

