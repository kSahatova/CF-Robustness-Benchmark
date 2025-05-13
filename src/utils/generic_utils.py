# pyright: reportMissingImports=false

import os
import yaml
import random
import numpy as np
from typing import Any, Tuple, List
from easydict import EasyDict

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Accuracy

import tensorflow as tf


DEFAULT_RANDOM_SEED = 2025


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# tensorflow random seed
def seed_tf(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)


# torch random seed
def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# basic + tensorflow + torch
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_tf(seed)
    seed_torch(seed)


def get_config(config_path: str):
    """Load the configuration file from the given path."""
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Convert the configuration to a dictionary
    config = EasyDict(config)
    return config


def load_model_weights(
    model: Any, framework: str = "torch", weights_path: str = "", **kwargs
):
    """Load pytorch model from the given weights path"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if framework == "torch":
            checkpoint = torch.load(
                    weights_path, weights_only=False,
                    map_location=torch.device(device)
                )
            
            model.load_state_dict(checkpoint["state_dict"])

        elif framework == "tf":
            # raise NotImplementedError("Check validity of the loading function here")
            model.load_weights(weights_path)  # kwargs["custom_objects"]
    
    except ValueError as e:
        print("Only pytorch or tensorflow models can loaded, check 'framework' argument which should be 'torch' or 'tf'. The following error occurred: ", e)
    except Exception as e:
        print(f"Error loading model weights: {e}")



def evaluate_classification_model(model, dataloader, num_classes):
    calc_metric = Accuracy(task="binary" if num_classes==2 else "multiclass", num_classes=num_classes)

    metric = 0
    for images, labels in dataloader:
        preds = torch.argmax(model(images), axis=1, keepdim=False)
        metric += calc_metric(preds, labels)

    print("Accuracy for the test dataset: {:.3%}".format(metric / len(dataloader)))


def compute_reconstruction_error(dataloader, autoencoder):
    """Compute the reconstruction error for a given autoencoder and data points."""
    # total_error = 0
    errors = []
    n_samples = 0
    for x, _ in dataloader:
        n_samples += x.shape[0]
        if not isinstance(x, np.ndarray):
            x = x.numpy()
        x_flat = x.reshape((x.shape[0], -1))
        preds = autoencoder.predict(x, verbose=0)
        preds_flat = preds.reshape((preds.shape[0], -1))

        batch_error = np.linalg.norm(x_flat - preds_flat, axis=1)
        # total_error += np.sum(batch_error)
        errors.append(batch_error)

    return np.asarray(errors)  # total_error


def format_metric(metric):
    """Return a formatted version of a metric, with the confidence interval."""
    return f"{metric.mean():.3f} ± {1.96 * metric.std() / np.sqrt(len(metric)):.3f}"


def extract_factual_instances(dataloader: DataLoader, init_class_idx: List[int]) -> Tuple[Tensor, Tensor]:
    """Extracts factual instances of the provided class index"""

    factuals_list = []
    labels_list = []
    for imgs, labels in dataloader:
        for class_ind in init_class_idx:
            ind = torch.where(labels == class_ind)[0]
            labels_list.append(labels[ind])
            factuals_list.append(imgs[ind])

    factuals_tensor = torch.concat(factuals_list)
    labels_tensor = torch.concat(labels_list)

    return factuals_tensor, labels_tensor


def filter_valid_factuals(factuals: Tensor, labels: Tensor, classifier: nn.Module, device: str="cpu") -> Tuple[Tensor, Tensor]:
    """Filters out instances that are invalidated by the given classifier"""

    classifier = classifier.to(device)
    predictions = torch.argmax(classifier(factuals.to(device)), axis=1).detach().cpu()
    valid_indices = np.where(predictions == labels)[0]
    factuals_tensor = factuals[valid_indices] 
    labels_tensor = labels[valid_indices]

    return factuals_tensor, labels_tensor
