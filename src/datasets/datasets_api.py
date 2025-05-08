# pyright: reportMissingImports=false

import numpy as np
from typing import Any, List
import albumentations as A
from sklearn.model_selection import train_test_split

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torch.utils.data import Dataset

from imblearn.under_sampling import RandomUnderSampler
from src.datasets import medmnist_corrected as medmnist


class AbstractDataset(Dataset):
    def __init__(
        self,
        ds_name: str = None,
        data_dir: str = "",
        increase_channels: bool = False,
        channels_first: bool = True,
    ):
        self.ds_name = ds_name
        self.data_dir = data_dir
        self.increase_channels = increase_channels
        self.channels_first = channels_first

    def _add_channels_dimension(self, image):
        """Adds extra dimension for channels since MedMNIST provides 3 dimensional array for images (BxHxW)"""
        try:
            if (len(image.shape) < 3) and isinstance(image, np.ndarray):
                return np.expand_dims(image, axis=-1)
        except Exception as e:
            print(e)
        else:
            return image

    def _increase_channels(self, image):
        """Increase the number of channels from 1 to 3"""
        if self.increase_channels:
            return np.tile(image, [1, 1, 3])
        else:
            return image

    def __len__(self):
        return len(self.data)

    def _load_data(self, data_dir, download):
        raise NotImplementedError("The method has not been implemented yet")

    def __getitem__(self, idx):
        raise NotImplementedError("The method has not been implemented yet")


class MedMNISTDataset(AbstractDataset):
    """
    A customization of the MedMNIST dataset that allows integration of advanced transformations
    """

    def __init__(
        self,
        ds_name: str = "dermamnist",
        data_dir: str = "",
        img_size: int = 224,
        download: bool = True,
        split: str = "train",
        transform: Any = None,
        increase_channels: bool = False,
        undersample: bool = True,
        channels_first: bool = True,
        classes: List[str] = ["all"],
    ):
        super().__init__(ds_name, data_dir, increase_channels, channels_first)

        self.img_size = img_size
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.increase_channels = increase_channels
        self.undersample = undersample
        self.channels_first = channels_first
        self.classes = classes

        self.data = self._load_data(data_dir, download)
        if self.undersample:
            self._undersample()

    def _load_data(self, data_dir, download):
        info = medmnist.INFO[self.ds_name]
        DataClass = getattr(medmnist, info["python_class"])

        try:
            dataset = DataClass(
                split=self.split, size=self.img_size, root=data_dir, download=download
            )

            if self.classes != ["all"]:
                class_encodings = {v: k for k, v in dataset.info["label"].items()}
                labels = []
                images = []
                for i, class_name in enumerate(self.classes):
                    class_ind = class_encodings[class_name]
                    indices, _ = np.where(dataset.labels == int(class_ind))
                    images.append(dataset.imgs[indices])
                    labels.append(np.full(dataset.labels[indices].shape, fill_value=i))

                dataset.imgs = np.concatenate(images)
                dataset.labels = np.concatenate(labels)
                # change the class info correspondingly
                dataset.info["label"] = {
                    str(i): class_name for i, class_name in enumerate(self.classes)
                }
                dataset.info["n_samples"] = {self.split: dataset.imgs.shape[0]}
                if len(self.classes) == 2:
                    dataset.info["task"] = "binary"
            return dataset
        except Exception as e:
            print(f"Error loading the dataset {self.ds_name}: {e}")
            raise

    def __getitem__(self, idx):
        """Returns an image and label for the given index"""
        label = self.data.labels[idx].squeeze(-1).astype(np.int64)

        metadata = {
            "bbox": [0.15, 0.3, 0.9, 0.4],
            "text": "R",
        }

        image = self.data.imgs[idx]
        augmented = self.transform(image=image, textimage_metadata=metadata)["image"]
        image = self._add_channels_dimension(augmented)
        image = self._increase_channels(image)
        image = np.transpose(image, axes=(2, 0, 1)) if self.channels_first else image

        return image, label

    def _undersample(self):
        """Undersample the dataset to balance the classes."""

        rus = RandomUnderSampler(sampling_strategy=0.7, random_state=0)

        reshaped_images = np.reshape(self.data.imgs, (self.data.imgs.shape[0], -1))
        images, labels = rus.fit_resample(reshaped_images, self.data.labels)
        images = images.reshape(images.shape[0], self.img_size, self.img_size)

        if len(labels.shape) < 2:
            labels = np.expand_dims(labels, axis=-1)

        self.data.imgs = images
        self.data.labels = labels

        self.data.info["n_samples"][self.split] = images.shape[0]


class CIFAR10Dataset(Dataset):
    # TODO: change the methods for data loading
    def __init__(
        self,
        ds_name: str = "cifar10",
        data_dir: str = "",
        size: int = 32,
        split: str = "train",
        transform: Any = None,
        download: bool = True,
        increase_channels: bool = False,
        undersample: bool = True,
        channels_first: bool = True,
    ):
        self.ds_name = ds_name
        self.data_dir = data_dir
        self.split = split
        self.img_size = size
        self.channels_first = channels_first
        self.transform = (
            A.Compose(
                [
                    A.Normalize(mean=0.0, std=1.0),
                ]
            )
            if transform is None
            else transform
        )
        self.undersample = undersample
        self.increase_channels = increase_channels

        self.data = self._load_data(data_dir, download)
        if channels_first:
            self.data.data = np.transpose(self.data.data, (0, 3, 1, 2))

    def _load_data(self, data_dir, download):
        if self.split == "test":
            # Load the official CIFAR-10 test set
            return CIFAR10(root=data_dir, train=False, download=download)
        else:
            # Load the full training set
            sample = CIFAR10(root=data_dir, train=True, download=download)
            X, y = sample.data, sample.targets

            # Split into train and val
            val_ratio = 0.2
            train_idx, val_idx = train_test_split(
                range(len(X)), test_size=val_ratio, random_state=42, stratify=y
            )

            indices = train_idx if self.split == "train" else val_idx
            sample.data = X[indices]
            sample.targets = [y[i] for i in indices]
            return sample

    def _train_val_split(self, data, targets):
        split_ratio = 0.2
        train_size = int((1 - split_ratio) * len(self.data.data))
        indices = np.random.shuffle(np.arange(len(self.data.data)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        if self.split == "train":
            data = data[train_indices]
            targets = targets[train_indices]
        elif self.split == "val":
            data = data[val_indices]
            targets = targets[val_indices]
        return data, targets

    def __getitem__(self, idx):
        label = self.data.targets[idx]
        image = self.data.data[idx]

        image = self.transform(image=image)["image"]
        # image = self._add_channels_dimension(augmented)
        # image = self._increase_channels(image)

        return image, label

    def __len__(self):
        return len(self.data)

    def _add_channels_dimension(
        self,
    ):
        raise NotImplementedError("The method has not been implemented yet")

    def _increase_channels(
        self,
    ):
        raise NotImplementedError("The method has not been implemented yet")


class MNISTDataset(AbstractDataset):
    def __init__(
        self,
        ds_name: str = "mnist",
        data_dir: str = "",
        img_size: int = 28,
        split: str = "train",
        transform: Any = None,
        download: bool = True,
        increase_channels: bool = False,
        undersample: bool = True,
        channels_first: bool = True,
        classes: List[str] = ["all"],
    ):
        self.ds_name = ds_name
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.channels_first = channels_first
        self.transform = (
            A.Compose([A.Normalize(mean=0.0, std=1.0)])
            if transform is None
            else transform
        )
        self.undersample = undersample
        self.increase_channels = increase_channels
        self.classes = classes
        self.data = self._load_data(data_dir, download)

    def _load_data(self, data_dir, download):
        if self.split == "test":
            # Load the official MNIST test set
            sample = MNIST(root=data_dir, train=False, download=download)
        else:
            # Load the full training set
            sample = MNIST(root=data_dir, train=True, download=download)
            X, y = sample.data, sample.targets

            # Split into train and val
            val_ratio = 0.2
            train_idx, val_idx = train_test_split(
                range(len(X)), test_size=val_ratio, random_state=42, stratify=y
            )

            indices = train_idx if self.split == "train" else val_idx
            sample.data = X[indices]
            sample.targets = [y[i] for i in indices]

        if not isinstance(sample.data, np.ndarray):
            sample.data = np.asarray(sample.data)
        if not isinstance(sample.targets, np.ndarray):
            sample.targets = np.asarray(sample.targets)

        if self.classes != ["all"]:
            sample.classes = {k: v for k, v in enumerate(self.classes)}
            targets = []
            images = []
            for i, class_ind in enumerate(self.classes):
                indices = np.where(sample.targets == class_ind)
                class_i_targets = np.full(
                    shape=sample.targets[indices].shape, fill_value=i
                )
                targets.append(class_i_targets)
                images.append(sample.data[indices])
            sample.targets = np.concatenate(targets)
            sample.data = np.concatenate(images)

        return sample

    def _train_val_split(self, data, targets):
        split_ratio = 0.2
        train_size = int((1 - split_ratio) * len(self.data.data))
        indices = np.random.shuffle(np.arange(len(self.data.data)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        if self.split == "train":
            data = data[train_indices]
            targets = targets[train_indices]
        elif self.split == "val":
            data = data[val_indices]
            targets = targets[val_indices]

        return data, targets

    def __getitem__(self, idx):
        label = self.data.targets[idx].astype(np.int64)
        image = self.data.data[idx]

        image = self.transform(image=image)["image"]
        image = self._add_channels_dimension(image)
        image = self._increase_channels(image)
        image = np.transpose(image, axes=(2, 0, 1)) if self.channels_first else image

        return image, label

    def __len__(self):
        return len(self.data)

    def _add_channels_dimension(self, image):
        """Adds extra dimension for channels since MedMNIST provides 3 dimensional array for images"""
        try:
            if (len(image.shape) < 3) and isinstance(image, np.ndarray):
                return np.expand_dims(image, axis=-1)
        except Exception as e:
            print(e)
        else:
            return image

    def _increase_channels(self, image):
        """Increase the number of channels from 1 to 3"""
        if self.increase_channels:
            return np.tile(image, [1, 1, 3])
        else:
            return image


class FashionMNISTDataset(AbstractDataset):
    def __init__(
        self,
        ds_name: str = "fashionmnist",
        data_dir: str = "",
        img_size: int = 28,
        split: str = "train",
        transform: Any = None,
        download: bool = True,
        increase_channels: bool = False,
        undersample: bool = True,
        channels_first: bool = True,
        classes: List[str] = ["all"],
    ):
        self.ds_name = ds_name
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.channels_first = channels_first
        self.transform = (
            A.Compose([A.Normalize(mean=0.0, std=1.0)])
            # transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        )
        self.undersample = undersample
        self.increase_channels = increase_channels
        self.classes = classes
        self.data = self._load_data(data_dir, download)

    def _load_data(self, data_dir, download):
        if self.split == "test":
            sample = FashionMNIST(root=data_dir, train=False, download=download)
        else:
            # Load the full training set
            sample = FashionMNIST(root=data_dir, train=True, download=download)
            X, y = sample.data, sample.targets

            # Split into train and val
            val_ratio = 0.2
            train_idx, val_idx = train_test_split(
                range(len(X)), test_size=val_ratio, random_state=42, stratify=y
            )

            indices = train_idx if self.split == "train" else val_idx
            sample.data = X[indices]
            sample.targets = [y[i] for i in indices]

        if not isinstance(sample.data, np.ndarray):
            sample.data = np.asarray(sample.data)
        if not isinstance(sample.targets, np.ndarray):
            sample.targets = np.asarray(sample.targets)

        if self.classes != ["all"]:
            sample.classes = {k: v for k, v in enumerate(self.classes)}
            targets = []
            images = []
            for i, class_ind in enumerate(self.classes):
                indices = np.where(sample.targets == class_ind)
                class_i_targets = np.full(
                    shape=sample.targets[indices].shape, fill_value=i
                )
                targets.append(class_i_targets)
                images.append(sample.data[indices])
            sample.targets = np.concatenate(targets)
            sample.data = np.concatenate(images)

        return sample

    def _train_val_split(self, data, targets):
        split_ratio = 0.2
        train_size = int((1 - split_ratio) * len(self.data.data))
        indices = np.random.shuffle(np.arange(len(self.data.data)))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        if self.split == "train":
            data = data[train_indices]
            targets = targets[train_indices]
        elif self.split == "val":
            data = data[val_indices]
            targets = targets[val_indices]

        return data, targets

    def __getitem__(self, idx):
        label = self.data.targets[idx].astype(np.int64)
        image = self.data.data[idx]

        image = self.transform(image=image)["image"]
        image = self._add_channels_dimension(image)
        image = self._increase_channels(image)
        image = np.transpose(image, axes=(2, 0, 1)) if self.channels_first else image

        return image, label

    def __len__(self):
        return len(self.data)

    def _add_channels_dimension(self, image):
        """Adds an extra dimension for channels"""
        try:
            if (len(image.shape) < 3) and isinstance(image, np.ndarray):
                return np.expand_dims(image, axis=-1)
        except Exception as e:
            print(e)
        else:
            return image

    def _increase_channels(self, image):
        """Increase the number of channels from 1 to 3"""
        if self.increase_channels:
            return np.tile(image, [1, 1, 3])
        else:
            return image
