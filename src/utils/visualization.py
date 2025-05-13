import os.path as osp
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Union


def plot_boxplot(data: Union[pd.DataFrame, List], 
                 mean_values: Optional[Union[List, pd.DataFrame, pd.Series]],
                 tick_labels: List[str], 
                 y_label: str,
                 colors: List,
                 title: str):
    
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
        colors = colors * len(bp['boxes'])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(y_label)
    ax.yaxis.grid(True)
    
    if 'mean_values' in locals():
        indent_x = 1.15
        for mean in mean_values:
            ax.text(indent_x, mean, str(mean), 
                    fontsize=11, color='black', backgroundcolor='lightgreen')
            indent_x += 1
    

    # Show the plot
    plt.tight_layout()
    plt.title(title)
    plt.show()