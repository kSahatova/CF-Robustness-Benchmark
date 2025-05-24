from . import evaluate
from . import helper
from . import models
from . import modules
from . import train
from . import args

import torch
from .modules import NLMappingConv
from .helper import (
    load_pretrained_classifier,
    load_pretrained_gan,
    load_pretrained_encoder,
)
from easydict import EasyDict
from src.utils import load_model_weights


class C3LTModel:
    def __init__(self, config: EasyDict):
        """
        Initialize the C3LT model with the given configuration.
        Args:
            config (EasyDict): Configuration object containing model parameters.
        """
        self.config = config

    def setup_model(self, mapper_weights_path: str = ""):
        """
        Sets up the C3LT model by loading the pretrained components.
        Args:
            mapper_weights_path (str): Path to the pretrained weights for the NLMappingConv module.
        """

        self.mapper = NLMappingConv(self.config.latent_dim).to(self.config.device)
        try:
            load_model_weights(
                self.mapper, framework="torch", weights_path=mapper_weights_path
            )
        except Exception as e:
            print(f"Error loading mapper weights: {e}")
            raise e
        self.classifier = load_pretrained_classifier(self.config)
        self.generator, _ = load_pretrained_gan(self.config)
        self.encoder = load_pretrained_encoder(self.config)

    def get_counterfactuals(self, factuals: torch.Tensor) -> torch.Tensor:
        """
        Generate counterfactuals for the given factuals using the pretrained parts of the C3LT model.
        Args:
            factuals (torch.Tensor): The input factuals to generate counterfactuals for.
        Returns:
            torch.Tensor: The generated counterfactuals.
        """

        embeddings = self.encoder(factuals)
        cf_embeddings = self.mapper(embeddings)
        cfes = self.generator(cf_embeddings)

        return cfes
