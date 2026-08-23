import copy
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100
from torchvision.transforms import functional as F

from utils import Batch


def find_project_root():
    """Walks upward from this file to the directory holding pyproject.toml or .git."""
    here = Path(__file__).resolve().parent

    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    raise RuntimeError(
        f"Could not find project root from {here}. "
        "Expected pyproject.toml or .git in one of the parent folders."
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class RotateTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return F.rotate(x, self.angle)


def _build_tinyimagenet_val(val_dir, transform):
    """Returns an ImageFolder over TinyImageNet val, restructuring it once.

    TinyImageNet ships val images flat in val/images/ with labels in
    val/val_annotations.txt; ImageFolder expects one subfolder per class.
    """
    structured_dir = os.path.join(val_dir, "structured")
    if os.path.isdir(structured_dir):
        return datasets.ImageFolder(root=structured_dir, transform=transform)

    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, class_id = parts[0], parts[1]
            class_dir = os.path.join(structured_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(val_dir, "images", fname)
            dst = os.path.join(class_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    return datasets.ImageFolder(root=structured_dir, transform=transform)


def download_tinyimagenet(data_dir=DATA_DIR):
    target = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.isdir(target):
        return
    import urllib.request
    import zipfile

    os.makedirs(data_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    print(f"Downloading TinyImageNet to {zip_path} ...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    os.remove(zip_path)


def load_data(dataset):
    if "mnist" in dataset:
        if dataset == "permutedmnist":
            full_dataset, test_dataset = torch.load(
                DATA_DIR / "PMNIST" / "mnist_permutations.pt"
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            full_dataset = datasets.MNIST(
                DATA_DIR, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                DATA_DIR, train=False, transform=transform
            )
    elif dataset.startswith("cifar100"):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        full_dataset = CIFAR100(
            root=DATA_DIR, train=True, transform=transform, download=True
        )
        test_dataset = CIFAR100(
            DATA_DIR, train=False, transform=transform, download=True
        )
    elif dataset.startswith("tinyimagenet"):
        download_tinyimagenet()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )
        full_dataset = datasets.ImageFolder(
            root=DATA_DIR / "tiny-imagenet-200" / "train", transform=transform
        )
        test_dataset = _build_tinyimagenet_val(
            val_dir=DATA_DIR / "tiny-imagenet-200" / "val", transform=transform
        )
    return full_dataset, test_dataset


def split_task_construction(dataset, task_labels, class_il=False):
    full_dataset, test_dataset = load_data(dataset)
    train_datasets, test_datasets = [], []
    for labels in task_labels:
        # Map each class to a label in {0, ..., n-1} consistently in train
        # and test. Class-IL keeps the original labels instead.
        new_labels = random.sample(range(len(labels)), len(labels))
        label_map = dict(zip(labels, new_labels))
        if class_il:
            label_map = dict(zip(labels, labels))
        train_datasets.append(
            create_split_dataset(full_dataset, labels, label_map)
        )
        test_datasets.append(
            create_split_dataset(test_dataset, labels, label_map)
        )
    return train_datasets, test_datasets


def create_split_dataset(dataset, labels, label_map):
    keep = np.isin(dataset.targets, labels)
    split_dataset = copy.deepcopy(dataset)
    if isinstance(split_dataset.targets, list):
        split_dataset.targets = torch.FloatTensor(split_dataset.targets)
    split_dataset.targets = split_dataset.targets[keep]
    split_dataset.targets = torch.from_numpy(
        np.array([label_map[int(label)] for label in split_dataset.targets])
    )
    if isinstance(split_dataset, datasets.ImageFolder):
        new_imgs = []
        for i, include in enumerate(keep):
            if include:
                img_path, old_label = split_dataset.imgs[i]
                new_imgs.append((img_path, label_map[int(old_label)]))
        split_dataset.imgs = new_imgs
        split_dataset.samples = new_imgs
    else:
        split_dataset.data = split_dataset.data[keep]
    return split_dataset


def rotated_task_construction(n_tasks):
    train_datasets, test_datasets = [], []
    rotation_angles = []
    min_angle, max_angle = 0.0, 180.0
    for t in range(n_tasks):
        task_min = 1.0 * t / n_tasks * (max_angle - min_angle) + min_angle
        task_max = (
            1.0 * (t + 1) / n_tasks * (max_angle - min_angle) + min_angle
        )
        rot = round(random.random() * (task_max - task_min) + task_min)

        transform = transforms.Compose(
            [
                RotateTransform(rot),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_datasets.append(
            datasets.MNIST(
                DATA_DIR, train=True, download=True, transform=transform
            )
        )
        test_datasets.append(
            datasets.MNIST(
                DATA_DIR, train=False, download=True, transform=transform
            )
        )
        rotation_angles.append(rot)
    return list(train_datasets), list(test_datasets), list(rotation_angles)


def _stack_to_device(items, device):
    # torch.stack on tensor items skips the numpy round trip; values are
    # identical either way.
    if torch.is_tensor(items[0]):
        return torch.stack(items).to(device).float()
    return torch.from_numpy(np.array(items)).to(device).float()


class BatchGenerator:
    def __init__(self, task_dataset, config_params):
        self.data = task_dataset
        self.config_params = config_params
        self.num_classes = config_params["n"]
        self.k = config_params["k"]
        self.q = 100

        self.images_by_class = defaultdict(list)
        if config_params["dataset"] == "permutedmnist":
            for img, c in zip(self.data[0], self.data[1]):
                self.images_by_class[int(c)].append(img)
        else:
            for img, c in task_dataset:
                self.images_by_class[int(c)].append(img)
        for c in self.images_by_class.keys():
            self.q = min(self.q, len(self.images_by_class[c]) - self.k)

    def get_batch(self, device=None):
        if device is None:
            device = self.config_params["device"]

        classes = list(self.images_by_class.keys())
        label_map = dict(zip(classes, classes))
        if self.config_params["method"] in ("maml", "proposed", "l2l"):
            # Meta-learners see each class under a random label in {0, ..., n-1}.
            labels = random.sample(range(self.num_classes), self.num_classes)
            label_map = dict(zip(classes, labels))

        # k support and q query examples from each class.
        x_sp, y_sp, x_qr, y_qr = [], [], [], []
        for c in classes:
            images = random.sample(self.images_by_class[c], self.k + self.q)
            x_sp += images[: self.k]
            y_sp += [label_map[c] for _ in range(self.k)]
            x_qr += images[self.k :]
            y_qr += [label_map[c] for _ in range(self.q)]

        x_sp, y_sp, x_qr, y_qr = [
            _stack_to_device(lst, device) for lst in [x_sp, y_sp, x_qr, y_qr]
        ]
        y_sp, y_qr = y_sp.long(), y_qr.long()

        return Batch(x_sp, y_sp, x_qr, y_qr)
