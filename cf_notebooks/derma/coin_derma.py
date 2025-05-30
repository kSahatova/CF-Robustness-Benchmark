import warnings
warnings.filterwarnings("ignore")
# expandable_segments allows the allocator to create a segment initially and then expand its size later when more memory is needed.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import tracemalloc
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

import torch

from src.datasets import DatasetBuilder
from src.utils import seed_everything, get_config, load_model_weights, evaluate_classification_model
from src.cf_methods.coin import CounterfactualCGAN, CounterfactualTrainer, CounterfactualInpaintingCGAN

seed_everything()



def main():
    config_dir = '/data/leuven/365/vsc36567/CF-Robustness-Benchmark/configs' #'D:\PycharmProjects\CF-Robustness-Benchmark\configs' #
    config_path = osp.join(config_dir, 'coin_derma.yaml') 
    config = get_config(config_path) 

    ds_builder = DatasetBuilder(config)
    ds_builder.setup()
    train_loader, val_loader, test_loader = ds_builder.get_dataloaders()

    cfcgan_inpainting = CounterfactualInpaintingCGAN(opt=config, img_size=config.data.img_size)
    trainer = CounterfactualTrainer(opt=config, model=cfcgan) # continue_path=continue_path)