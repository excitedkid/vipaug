import torch
from torchvision import transforms

from datasets.VIP import VIPAug

def train_transforms(_transforms, kernel, vital, nonvital, dataset_name, fractal_images):
    transforms_list = []
    if 'vipaug' == _transforms:
        print('vipaug', _transforms)
        transforms_list.extend([
            transforms.RandomApply([VIPAug(kernel, vital, nonvital, dataset_name, fractal_images)], p=1.0),
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transforms_list.extend([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    return transforms_list


def test_transforms():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return test_transform