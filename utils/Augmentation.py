from torch.utils.data._utils.collate import default_collate
from datasets.transforms import *

class SplitLabel(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return self.worker(img_group), label



def train_augmentation(input_size, flip=True):
    if flip:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            GroupRandomHorizontalFlip(is_flow=False)])
    else:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            # GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip_sth()])


def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    normalize = GroupNormalize(input_mean, input_std)
    if 'something' in config.data.dataset:
        groupscale = GroupScale((256, 320))
    else:
        groupscale = GroupScale(int(scale_size))


    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize])

    if training:
        train_aug = train_augmentation(
            config.data.input_size,
            flip=False if 'something' in config.data.dataset else True)

        unique = torchvision.transforms.Compose([
            groupscale,
            train_aug,
            GroupRandomGrayscale(p=0 if 'something' in config.data.dataset else 0.2),
        ])
            
        return torchvision.transforms.Compose([unique, common])

    else:
        unique = torchvision.transforms.Compose([
            groupscale,
            GroupCenterCrop(config.data.input_size)])
        return torchvision.transforms.Compose([unique, common])



def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    inputs, labels = (
        default_collate(inputs),
        default_collate(labels),
    )
    return inputs, labels