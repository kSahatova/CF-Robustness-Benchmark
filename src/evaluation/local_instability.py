import numpy as np

from torchvision import transforms

# from torch.distributions.multivariate_normal import MultivariateNormal
from numpy.random import multivariate_normal, uniform
from torchmetrics.image import StructuralSimilarityIndexMeasure


# def perturb_sample(input_images, n_samples=1, type='uniform', epsilon=0.01):
#     batch_size, channels, height, width = input_images.shape
#     input_images = np.tile(input_images, reps=[n_samples, 1, 1, 1])

#     if type == 'normal':
#         # Define the mean and covariance for the multivariate normal distribution
#         mean = np.zeros(channels * height * width)  # Zero mean for all pixels
#         covariance = np.eye(channels * height * width) # Diagonal covariance

#         # Create the MultivariateNormal distribution
#         noise = multivariate_normal(mean, covariance, size=(batch_size * n_samples,))
#         noise = noise.reshape(batch_size * n_samples, channels, height, width)

#     elif type == 'uniform':
#         noise = uniform(-epsilon, epsilon, size=(batch_size, n_samples, channels, height, width))

#     # Perturb the input images with the generated noise
#     perturbed_images = input_images + noise

#     return perturbed_images


def perturb_sample(
    input_images, n_samples=1, type="uniform", epsilon=None, channels_first=True, std=0.1
):
    """Generate perturbed samples around the input images.
    Args:
        input_images (numpy array): Input images to be perturbed.
        n_samples (int): Number of perturbed samples to generate per input image.
        type (str): Type of noise to add ('normal' or 'uniform').
        epsilon (float): Magnitude of the noise.
        channels_first (bool): Whether the input images have channels first format.
        std (float): Standard deviation for normal noise.
    Returns:
        numpy array: Perturbed samples of shape (batch_size, n_samples, height, width, channels) or (batch_size, n_samples, channels, height, width).
    """
    data = input_images
    if len(input_images.shape) == 3:
        data = np.expand_dims(input_images, axis=0)
    if channels_first:
        batch_size, height, width, channels = data.shape
        result_shape = (batch_size, n_samples, height, width, channels)
    else:
        batch_size, channels, height, width = data.shape
        result_shape = (batch_size, n_samples, channels, height, width)
    data = np.expand_dims(data, axis=1)
    data = np.tile(data, reps=[1, n_samples, 1, 1, 1])

    if type == "normal":
        # Define the mean and covariance for the multivariate normal distribution
        mean = np.zeros(channels * height * width)  # Zero mean for all pixels
        covariance = np.eye(channels * height * width) * std**2  # Diagonal covariance

        # Create the MultivariateNormal distribution
        noise = multivariate_normal(mean, covariance, size=(batch_size, n_samples))
        noise = noise.reshape(*result_shape)
        if epsilon is not None:
            noise = np.clip(noise, -epsilon, epsilon)

    elif type == "uniform":
        noise = uniform(-epsilon, epsilon, size=result_shape)

    # Perturb the input images with the generated noise and normalize to [0, 1]
    perturbed_data = data + noise
    perturbed_data = np.clip(perturbed_data, a_min=0, a_max=1)

    return perturbed_data


def calculate_sparsity(factuals, counterfactuals) -> np.ndarray:
    """
    Calculates sparsity between original image and counterfactual image
    Args:
        factuals : Original image of the size (h x w X c)
        counterfactuals : Counterfactual explanation of the same size (h x w x c)
    Returns :
        the sum of squarred errors between the provided inputs
    """

    return np.linalg.norm(
        factuals - counterfactuals, ord=1
    )  # sum(abs(factual - counterfcatual))


def calculate_ssim(cf, cf_pert):
    normalize = transforms.Compose(
        [transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))]
    )
    cf_norm = normalize(cf)
    cf_pert_norm = normalize(cf_pert)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(cf_norm, cf_pert_norm)
