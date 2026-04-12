import torch.nn as nn


class DermaMNISTFeatureExtractor(nn.Module):
    """
    Feature extractor for DermaMNIST (64x64, 3-channel RGB) classifiers.

    modules. This works for any classifier architecture:
      - CNNtorch / SimpleCNNtorch: pass layer names like ``("main.0", "main.1")``
        to capture after each conv block, or ``("main",)`` for the full feature
        map before the linear head.
      - ResNet50 (via ``build_resnet50``): pass ``("layer1", "layer2", ...)`` to
        capture after each residual stage.

    Args:
        model (nn.Module): The pre-trained classifier to extract features from.
        extracted_layers (sequence of str): Names of submodules (as returned by
            ``model.named_modules()``) whose outputs should be captured.

    Example::

        featex = DermaMNISTFeatureExtractor(classifier, ("layer1", "layer2"))
        features = featex(imgs)   # list of tensors, one per extracted layer
    """

    def __init__(self, model, extracted_layers):
        super(DermaMNISTFeatureExtractor, self).__init__()
        self.model = model
        self.extracted_layers = list(extracted_layers)
        self._features = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.extracted_layers:
                self._hooks.append(
                    module.register_forward_hook(
                        lambda mod, inp, out, n=name: self._features.__setitem__(n, out)
                    )
                )

    def forward(self, x):
        self._features = {}
        try:
            self.model(x)
        except Exception as e:
            print(f'DermaMNISTFeatureExtractor: forward pass error: {e}')
        return [self._features[layer] for layer in self.extracted_layers
                if layer in self._features]

    def remove_hooks(self):
        """Remove all registered forward hooks. Call when done to avoid leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


class LMapping(nn.Module):
    def __init__(self, latent_dim):
        super(LMapping, self).__init__()

        self.map = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=True))

    def forward(self, x):
        return self.map(x)


class NLMappingFC(nn.Module):
    def __init__(self, latent_dim):
        super(NLMappingFC, self).__init__()  # Batch, Latent, 1, 1

        self.T1 = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=True),
                                nn.ReLU(),
                                # nn.Dropout(0.25)
                                )
        self.T2 = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=True))

    def forward(self, x):
        return self.T2(self.T1(x))


class NLMappingConv(nn.Module):
    def __init__(self, latent_dim):
        super(NLMappingConv, self).__init__()  # Batch, Latent, 1, 1

        self.T1 = nn.Sequential(nn.Conv2d(latent_dim, latent_dim, 1, 1, 0, bias=True),
                                nn.ReLU(),
                                # nn.Dropout(0.25)
                                )
        self.T2 = nn.Sequential(nn.Conv2d(latent_dim, latent_dim, 1, 1, 0, bias=True))

    def forward(self, x):
        return self.T2(self.T1(x))


class MNISTFeatureExtractor(nn.Module):
    def __init__(self, model, extracted_layers):
        super(MNISTFeatureExtractor, self).__init__()
        self.model = model
        self.extracted_layers = extracted_layers
        self.stop_at = len(extracted_layers)

    # def forward(self, x):
    #     outputs = []
    #     count = 0
    #     x_prev = x
    #     for name, module in self.model._modules.items():
    #         x = module[:2](x_prev)  # gets the ReLU layer
    #         x_prev = module[2:](x)
    #         if name in self.extracted_layers:
    #             count += 1
    #             outputs.append(x)
    #             if count == self.stop_at:
    #                 return outputs
    #     return outputs

    def forward(self, x):
        outputs = []
        count = 0
        for name, module in self.model.named_modules():
            try:
                if name in self.extracted_layers and isinstance(module, nn.Sequential):
                    for submodule in module:
                        x = submodule(x)  # gets the ReLU layer
                        if isinstance(submodule, nn.Conv2d):
                            count += 1
                            outputs.append(x)
                            if count == self.stop_at:
                                return outputs
            except Exception as e:
                print(f'Could not extract the features, because of the the following error : \n{e}')
        return outputs


