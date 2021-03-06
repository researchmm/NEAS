import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from lib.dataset.cifar_utils import SubsetDistributedSampler
from lib.dataset.cifar_utils import CIFAR10Policy, Cutout


def data_transforms_cifar(config, cutout=False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if config.aa:
        train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(), CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def get_search_datasets(config):

    dataset = config.dataset.lower()
    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'cifar100':
        dset_cls = dset.CIFAR100
        n_classes = 100
    else:
        raise Exception("Not support dataset!")

    train_transform, valid_transform = data_transforms_cifar(config, cutout=False)
    train_data = dset_cls(root=config.data_dir, train=True, download=True, transform=train_transform)
    test_data = dset_cls(root=config.data_dir, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split_mid = int(np.floor(0.5 * num_train))

    train_sampler = SubsetDistributedSampler(train_data, indices[:split_mid])
    valid_sampler = SubsetDistributedSampler(train_data, indices[split_mid:num_train])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config.workers)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=config.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]

def get_augment_datasets(config):

    dset_cls = dset.CIFAR10


    train_transform, valid_transform = data_transforms_cifar(config, cutout=True)
    train_data = dset_cls(root=config.data, train=True, download=True, transform=train_transform)
    test_data = dset_cls(root=config.data, train=False, download=True, transform=valid_transform)
    train_sampler, test_sampler = None, None
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    #     else:
    #         # This will add extra duplicate entries to result in equal num
    #         # of samples per-process, will slightly alter validation results
    #         sampler = OrderedDistributedSampler(dataset)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size,
        sampler=test_sampler,
        pin_memory=True, num_workers=config.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]