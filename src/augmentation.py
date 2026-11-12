"""TrivialAugment + Cutout pipeline used for V4 PreAct training.

Extracted verbatim from nyu_preact_v3/core.py. Do not modify; changing these
functions changes the training recipe and breaks checkpoint parity.
"""
import torch
from torchvision import transforms as T

MEAN = (.4914, .4822, .4465)
STD = (.247, .2435, .2616)


class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        k = self.size
        y, z = torch.randint(0, 33 - k, (2,)).tolist()
        x = x.clone()
        x[:, y:y + k, z:z + k] = 0
        return x


def transform(training=False, cfg=None):
    ops = []
    if training:
        ops += [
            T.TrivialAugmentWide(),
            T.RandomHorizontalFlip(.5),
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
        ]
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if training:
        ops += [Cutout(cfg['cutout'])]
    return T.Compose(ops)
