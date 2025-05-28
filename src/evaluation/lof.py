import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import DataLoader
from typing import Union, List

from src.utils import extract_factual_instances


def estimate_anom_cfe_in_target_class(n_neighbors: int, target_class_ind: List[int], threshold: int,
                                    test_dataloader: DataLoader, cfes: np.ndarray):
    """ Estimates how anomaluous the generated CFEs comparing to the target class distribution
    Args: 
        n_neighbors (int): a number of neighbors around the given point to consider it an inlier
        target_class_ind (List[int]): a target class index to extract the corresponding factual instances
        threshold (int): a threshold to filter out anomaluous points
        test_loader (torch.data.utils.DataLoader): a dataloader for factuals extraction
        cfes (numpy.ndarray): counterfactual explanations
    Returns: 
        (anomaly_scores, anomaly_classes) (tuple(ndarray, ndarray)): scores and classes that label each point as 
        an outlier with -1 or an inlier with 1
        """

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, p=1)
    factuals, labels = extract_factual_instances(test_dataloader, init_class_idx=target_class_ind) 
    factuals_array = factuals.view(factuals.shape[0], -1).numpy()
    lof.fit(factuals_array)

    cfes_array = cfes.reshape(cfes.shape[0], -1)
    anomaly_scores = lof.score_samples(cfes_array)
    anomaly_classes = np.where(anomaly_scores < -1.2, -1, 1)

    return (anomaly_scores, anomaly_classes)  