import os 
import torch
import time
import logging
import datetime
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from typing import Union, Dict, List
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor



@torch.no_grad()
def posterior2bin(posterior_pred: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Given classifier predictions in range [0; 1] and the number of condition bins, returns the condition labels.

    Args:
        posterior_pred (torch.Tensor): classifier predictions
        num_bins (int): number of conditions

    Returns:
        torch.Tensor: resulting condition labels
    """
    posterior_pred = posterior_pred.cpu().numpy()
    bin_step = 1 / num_bins
    bins = np.arange(0, 1, bin_step)
    bin_ids = np.digitize(posterior_pred, bins=bins) - 1
    return Variable(LongTensor(bin_ids), requires_grad=False)


def grad_norm(module: torch.nn.Module, norm_type: Union[float, int, str]=2, group_separator: str = "/") -> Dict[str, float]:
    """Compute each parameter's gradient's Frobenius norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.
    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"grad_{norm_type}_norm{group_separator}{name}": p.grad.data.norm(norm_type)
        for name, p in module.named_parameters()
        if p.grad is not None
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type)
        norms[f"grad_{norm_type}_norm_total"] = total_norm
    return norms


class AvgMeter:
    def __init__(self) -> None:
        self.reset()

    def update(self, new_data):
        for k, v in new_data.items():
            self.data[k] += v
        self.steps += 1
    
    def average(self):
        return {k: v / self.steps for k, v in self.data.items()}
    
    def reset(self):
        self.data = defaultdict(int)
        self.steps = 0


def save_model(config:dict, model:torch.nn.Module, optimizers:List[torch.nn.Module], 
               current_step:int, epoch:int, checkpoint_dir:Path, **kwargs) -> Path:
    state = {
        'config': config,
        'model': model.state_dict(),
        'optimizers': [opt.state_dict() for opt in optimizers],
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime('%B %d, %Y'),
        **kwargs,
    }
    checkpoint_path = checkpoint_dir / f'checkpoint_{epoch}.pth'
    torch.save(state, checkpoint_path)
    return checkpoint_path


def get_experiment_folder_path(root_path, model_name, experiment_name='exp'):
    """Get an experiment folder path with the current date and time"""
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    output_folder = os.path.join(root_path, model_name + "-" + date_str + "-" + experiment_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def setup_logger(name, log_file=None, message_format='[%(asctime)s|%(levelname)s] - %(message)s', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(message_format, datefmt=r'%Y-%m-%d %H:%M:%S')
    
    if log_file:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


class Logger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.logger = setup_logger(__name__, self.log_dir / 'logs.log')
        self.logger.info(f"================ Session ({time.strftime('%c')}) ================")
        self.logger.info(f'Logging directory: {self.log_dir}')

    def log(self, data, step, phase='train'):
        for key, value in data.items():
            self.writer.add_scalar(f'{phase}/{key}', value, step)

    def info(self, msg, *args, **kwags):
        return self.logger.info(msg, *args, **kwags)

    def reset(self):
        for p in self.log_dir.glob('events*'):
            p.unlink(True)

    def __del__(self):
        self.writer.flush()


def confmat_vis_img(
    batch_masks_gt:torch.Tensor, 
    batch_masks_pred:torch.Tensor,
    normalized:bool=False
):
    B, _, H, W = batch_masks_gt.shape
    tp, fp, fn = _tp_fp_fn(batch_masks_pred.squeeze(1), batch_masks_gt.squeeze(1))
    vis = torch.zeros((B, H, W, 3)).byte()
    for i in range(B):
        vis[i][tp[i]] = torch.tensor((50, 168, 82)).byte().reshape(1, 1, 3)  # green
        vis[i][fp[i]] = torch.tensor((235, 64, 52)).byte().reshape(1, 1, 3)  # red
        vis[i][fn[i]] = torch.tensor((207, 207, 31)).byte().reshape(1, 1, 3) # yellow
    vis = vis.to(batch_masks_gt.device)
    return vis if not normalized else vis.div(255)