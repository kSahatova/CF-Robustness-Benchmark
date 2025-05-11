import numpy as np

from torchvision import transforms

# from torch.distributions.multivariate_normal import MultivariateNormal
from numpy.random import multivariate_normal
from torchmetrics.image import StructuralSimilarityIndexMeasure


def perturb_sample(input_images, n_samples = 1, noise_magnitude = 0.01):
    batch_size, channels, height, width = input_images.shape
    input_images = np.tile(input_images, reps=[n_samples, 1, 1, 1])

    # Define the mean and covariance for the multivariate normal distribution
    mean = np.zeros(channels * height * width)  # Zero mean for all pixels
    covariance = np.eye(channels * height * width) * noise_magnitude  # Diagonal covariance

    # Create the MultivariateNormal distribution
    noise = multivariate_normal(mean, covariance, size=(batch_size * n_samples,))
    noise = noise.reshape(batch_size * n_samples, channels, height, width)

    # Perturb the input images with the generated noise
    perturbed_images = input_images + noise

    return perturbed_images


def calculate_sparsity(factual, counterfactual) -> np.ndarray:
    """
    Calculates sparsity between original image and counterfactual image
    factual : Original image of the size (h x w X c) 
    counterfactual : Counterfactual explanation of the same size (h x w x c) 
    Returns :
        the sum of squarred errors between the provided inputs
    """
    
    return np.linalg.norm(factual - counterfactual, ord=1) #sum(abs(factual - counterfcatual))


def calculate_ssim(cf, cf_pert):
    normalize = transforms.Compose([transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))])
    cf_norm = normalize(cf)
    cf_pert_norm = normalize(cf_pert)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(cf_norm, cf_pert_norm)