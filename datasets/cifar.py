import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from datasets.transforms import train_transforms, test_transforms
class CIFARC(CIFAR10):
    def __init__(
            self,
            root,
            key = 'zoom_blur',
            transform = None,
            target_transform = None,
    ):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        data_path = os.path.join(root, key+'.npy')
        labels_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(labels_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10D(object):
    def __init__(self,  kernel, vital, nonvital, dataroot='', dataroot_c='', num_workers=4, batch_size=128, _transforms='', _eval='none', fractal_images=''):
        dataset_name = 'cifar10'
        transforms_list = train_transforms(_transforms, kernel, vital, nonvital, dataset_name, fractal_images)
        train_transform = transforms.Compose(transforms_list)
        test_transform = test_transforms()

        data_root = os.path.join(dataroot, 'cifar10')

        trainset = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        
        testset = CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        self.num_classes = 10

        if _eval == 'eval':
            self.corruption_loaders = dict()
            self.corruption_keys = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                            'brightness', 'contrast', 'elastic_transform', 'pixelate',
                            'jpeg_compression']

            data_root = os.path.join(dataroot_c, 'CIFAR-10-C')
            for key in self.corruption_keys:
                corruption_set = CIFARC(root=data_root, key=key, transform=test_transform)
                corruption_loader = torch.utils.data.DataLoader(
                    corruption_set, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )
                self.corruption_loaders[key] = corruption_loader

        self.normalize = transforms.Compose([
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                ])


class CIFAR100D(object):
    def __init__(self, kernel, vital, nonvital, dataroot='', dataroot_c='', num_workers=4, batch_size=128, _transforms='', _eval='none', fractal_images=''):
        dataset_name = 'cifar100'
        transforms_list = train_transforms(_transforms, kernel, vital, nonvital, dataset_name, fractal_images)
        train_transform = transforms.Compose(transforms_list)
        test_transform = test_transforms()

        data_root = os.path.join(dataroot, 'cifar100')

        trainset = CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        
        testset = CIFAR100(root=data_root, train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        self.num_classes = 100

        if _eval == 'eval':
            self.corruption_loaders = dict()
            self.corruption_keys = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                            'brightness', 'contrast', 'elastic_transform', 'pixelate',
                            'jpeg_compression']

            data_root = os.path.join(dataroot_c, 'CIFAR-100-C')
            for key in self.corruption_keys:
                corruption_set = CIFARC(root=data_root, key=key, transform=test_transform)
                corruption_loader = torch.utils.data.DataLoader(
                    corruption_set, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                )
                self.corruption_loaders[key] = corruption_loader

        self.normalize = transforms.Compose([
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ])