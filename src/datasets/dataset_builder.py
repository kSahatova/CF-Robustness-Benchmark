# pyright: reportMissingImports=false

from easydict import EasyDict
from torch.utils.data import DataLoader

from src.datasets import datasets_api
from src.datasets.augmentations import AUGMENTATIONS


class DatasetBuilder:
    """Builds the dataset using the provided configuration object"""

    def __init__(self, config: EasyDict):
        self.class_name = config.data.class_name
        self.args = config.data
        self.img_size = config.data.img_size
        self.batch_size = config.batch_size
        self.data_dir = config.data_dir
        self.class_encodings = {
            i: cls_name for i, cls_name in enumerate(config.data.classes)
        }

    def _load_dataset(self, split):
        """Loads the dataset using the provided class and configuration arguments."""
        try:
            dataset_cls = getattr(datasets_api, self.class_name)
            dataset = dataset_cls(
                ds_name=self.args.name,
                split=split,
                data_dir=self.data_dir,
                img_size=self.img_size,
                download=self.args.download,
                # TODO: create separate augmentations for train, val, and test
                transform=AUGMENTATIONS["standard"],
                increase_channels=self.args.increase_channels,
                undersample=self.args.undersample_flag,
                channels_first=self.args.channels_first,
                classes=self.args.classes,
            )
            return dataset
        except AttributeError as e:
            raise ValueError(
                f"Failed to load the dataset {self.class_name} due to the error: {e}"
            )

    def setup(self):
        """Sets up the dataset for training, validation, and test."""
        self.train_dataset = self._load_dataset(split="train")
        self.val_dataset = self._load_dataset(split="val")
        self.test_dataset = self._load_dataset(split="test")

    def get_dataloaders(self):
        """Returns the dataloaders for training, validation, and test."""
        return (
            DataLoader(
                dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
            ),
            DataLoader(
                dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
            ),
            DataLoader(
                dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False
            ),
        )


