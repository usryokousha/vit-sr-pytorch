import torch
import sys
from torch.utils.data import Dataset
from torchvision.transforms import AugMix
from torch import Tensor

import torch.nn.functional as F

from typing import Tuple, Dict, Optional, List, Union

VALID_AUGMENTATIONS = ["TranslateX", 
                       "TranslateY", 
                       "Contrast", 
                       "Brightness", 
                       "Sharpness", 
                       "Posterize", 
                       "Solarize", 
                       "Equalize", 
                       "AutoContrast"]

class CustomAugMix(AugMix):
    def __init__(self, jsd=True, *args, **kwargs):
        self.jsd = jsd
        super().__init__(*args, **kwargs)
    
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        s = {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Posterize": (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if self.all_ops:
            s.update(
                {
                    "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                    "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                }
            )
            
        for k in list(s.keys()):
            if k not in VALID_AUGMENTATIONS:
                del s[k]
        return s
      
    def forward(self, orig_img: Tensor) -> Tensor:
        if self.jsd:
            orig = orig_img.clone()
            aug1 = super().forward(orig_img)
            aug2 = super().forward(orig_img)
            return [orig, aug1, aug2]
        else:
            return super().forward(orig_img)
          
class JensenShannonLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, 
                weight: Optional[Tensor] = None, 
                size_average=None, 
                ignore_index: int = -100, 
                reduce=None, reduction: str = 'mean', 
                label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
       
    def jsd_loss(self, y_pred_clean, y_pred_aug1, y_pred_aug2) -> Tensor:
        p_clean = F.softmax(y_pred_clean, dim=1)
        p_aug1 = F.softmax(y_pred_aug1, dim=1)
        p_aug2 = F.softmax(y_pred_aug2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0)
        loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.0
        return loss
        
    def forward(self, input: Union[Tensor, List[Tensor]], target: Tensor) -> Tensor:
        if isinstance(input, list):
            return super().forward(input[0], target) + 12 * self.jsd_loss(input[0], input[1], input[2])
        else:
            return super().forward(input, target)
        
def bash_to_win32_path(path: str) -> str:
    """Changes path drive designation from /a/... to a:/..."""
    if sys.platform == "win32":
        path = path.replace("/", "\\")
        path = path[0] + ":" + path[1:]
    return path
      
