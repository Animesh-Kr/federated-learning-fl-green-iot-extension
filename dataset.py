# =============================================================================
# dataset.py
# CIFAR-10 dataset loading and federated partitioning.
#
# UPGRADE from group baseline: switched from MNIST to CIFAR-10.
# CIFAR-10 is a harder, more realistic benchmark for IoT FL systems —
# colour images (3-channel) with 10 object classes.
#
# Partition: IID random split so each client gets an equal share.
# Proper CIFAR-10 normalisation applied (per-channel mean/std).
# =============================================================================

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CIFAR-10 per-channel statistics
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def load_datasets(num_clients: int, seed: int = 42):
    """
    Partition CIFAR-10 training set into `num_clients` equal IID shards.

    Args:
        num_clients : Number of FL clients (e.g. 50).
        seed        : RNG seed for reproducible shuffling.

    Returns:
        trainloaders : list[DataLoader] — one per client
        testloader   : DataLoader       — full test set
    """
    rng = np.random.default_rng(seed)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),          # data augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])

    trainset = datasets.CIFAR10("./data", train=True,  download=True, transform=transform_train)
    testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    idx = np.arange(len(trainset))
    rng.shuffle(idx)
    chunks = np.array_split(idx, num_clients)

    trainloaders = [
        DataLoader(Subset(trainset, c.tolist()), batch_size=32,
                   shuffle=True, num_workers=0, pin_memory=False)
        for c in chunks
    ]
    testloader = DataLoader(testset, batch_size=64, shuffle=False,
                            num_workers=0, pin_memory=False)

    return trainloaders, testloader
