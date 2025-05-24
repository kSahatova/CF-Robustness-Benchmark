import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Union

import torch
from torch import Tensor


def plot_boxplot(
    data: Union[pd.DataFrame, List],
    mean_values: Optional[Union[List, pd.DataFrame, pd.Series]],
    tick_labels: List[str],
    y_label: str,
    colors: List,
    title: str,
):
    """
    Visualizes the results of evaluation metrics in the form of a boxplot
    data : Dataframe of size (n x m) that contains the results of evaluation,
      where n is a number of estimations and m is a number of metrics
    tick_labels : a list of labels to be assigned for each boxplot
    colors : a list of specified colors for each metric
    """

    _, ax = plt.subplots()
    bp = ax.boxplot(data, patch_artist=True, tick_labels=tick_labels, showmeans=True)

    # Customize boxplot colors
    if len(colors) == 1:
        colors = colors * len(bp["boxes"])

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(y_label)
    ax.yaxis.grid(True)

    if "mean_values" in locals():
        indent_x = 1.15
        for mean in mean_values:
            ax.text(
                indent_x,
                mean,
                str(mean),
                fontsize=11,
                color="black",
                backgroundcolor="lightgreen",
            )
            indent_x += 1

    # Show the plot
    plt.tight_layout()
    plt.title(title)
    plt.show()


"""
def plot_perturbed_factuals_and_cf():
    ind = 1

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            noise = noise_levels[j]
            if i == 0: 
                pert_image = factuals_pert[noise][ind].squeeze().squeeze() #perturb_sample(factuals_tensor[ind].unsqueeze(0), noise_magnitude=noise).squeeze().squeeze()
                axs[i, j].imshow(pert_image, cmap='gray')
                axs[i, j].set_title('$\\epsilon$='+str(noise)) 
                axs[i, j].axis('off')
            else:
                pert_sample = torch.Tensor(factuals_pert[noise]).to(device)
                latent_code_pert = encoder(pert_sample)
                mapping_pert = g(latent_code_pert)
                cfes_pert_sample = gen(mapping_pert)
                axs[i, j].imshow(cfes_pert_sample[ind].detach().cpu().numpy().transpose(1, 2, 0), cmap='gray')
                axs[i, j].axis('off')"""


def plot_perturbed_factuals_and_cf(
    initial_factual: Tensor,
    factuals_pert: List[Tensor],
    initial_cfe: Tensor,
    cfes_pert: List[Tensor],
    noise_levels: Optional[List[float]] = None,
) -> None:
    """
    Plots perturbed factuals and corresponding CFEs for different noise levels.
    Args:
        initial_factual (Tensor): Initial factual instance.
        factuals_pert (List[Tensor]): List of perturbed factual instances.
        initial_cfe (Tensor): Counterfactual explanation of the initial factual instance.
        cfes_pert (List[Tensor]): List of counterfactual explanations generated from the perturbed factual instances.
        noise_levels (List[float]): List of noise levels used for perturbation.
    Returns:
        None
    """
    plt.style.use("seaborn-v0_8")
    factuals = [initial_factual] + factuals_pert
    cfes = [initial_cfe] + cfes_pert

    n_rows = 2
    n_cols = len(factuals)

    _, axs = plt.subplots(n_rows, n_cols, figsize=(10, 4), sharex=True, sharey=True)
    for j in range(n_cols):
        axs[0, j].imshow(factuals[j].squeeze().squeeze() , cmap="gray")
        if noise_levels and j > 0:
            noise = noise_levels[j-1]
            axs[0, j].set_title("$\\epsilon=$" + str(noise))
        axs[0, j].axis("off")

        axs[1, j].imshow(cfes[j].squeeze().detach().cpu(), cmap="gray")
        axs[1, j].axis("off")
        axs[1, j].set_ylabel("CFE /wo noise")
        
        if j == 0:
            axs[0, j].set_ylabel('Factuasl')
            axs[1, j].set_ylabel('CFEs')

    plt.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.show()
