from copy import deepcopy
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.utils import save_image
from tqdm import tqdm

from src.cf_methods.coin.model import CounterfactualCGAN
from src.cf_methods.coin.utils import AvgMeter, save_model
from src.cf_methods.coin.trainers import BaseTrainer



class CounterfactualTrainer(BaseTrainer):
    def __init__(self, opt: edict, model: CounterfactualCGAN, continue_path: str = None) -> None:
        super().__init__(opt, model, continue_path)
        self.cf_vis_dir_train = self.logging_dir / 'counterfactuals/train'
        self.cf_vis_dir_train.mkdir(exist_ok=True, parents=True)
        
        self.cf_vis_dir_val = self.logging_dir / 'counterfactuals/val'
        self.cf_vis_dir_val.mkdir(exist_ok=True)

        self.compute_norms = self.opt.get('compute_norms', False)
        # 64, 192, 768, 2048
        self.fid_features = opt.get('fid_features', 768)
        self.val_fid = FrechetInceptionDistance(self.fid_features, normalize=True).to(self.device)
        
        self.cf_gt_seg_mask_idx = opt.get('cf_gt_seg_mask_idx', -1)
        self.cf_threshold = opt.get('cf_threshold', 0.25)
        self.val_iou_xc = BinaryJaccardIndex(self.cf_threshold).to(self.device)
        self.val_iou_xfx = BinaryJaccardIndex(self.cf_threshold).to(self.device)

    def restore_state(self):
        latest_ckpt = max(self.ckpt_dir.glob('*.pth'), key=lambda p: int(p.name.replace('.pth', '').split('_')[1]))
        load_ckpt = latest_ckpt if self.ckpt_name is None else (self.ckpt_dir / self.ckpt_name)
        state = torch.load(load_ckpt, weights_only=False)
        self.batches_done = state['step']
        self.current_epoch = state['epoch']
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer_G.load_state_dict(state['optimizers'][0])
        self.model.optimizer_D.load_state_dict(state['optimizers'][1])
        self.logger.info(f"Restored checkpoint {load_ckpt} ({state['date']})")

    def save_state(self) -> str:
        return save_model(self.opt, self.model, (self.model.optimizer_G, self.model.optimizer_D), self.batches_done, self.current_epoch, self.ckpt_dir)

    def training_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()

        stats = AvgMeter()
        epoch_steps = self.opt.get('epoch_steps')
        with tqdm(enumerate(loader), desc=f'Training epoch {self.current_epoch}', leave=False, total=epoch_steps or len(loader)) as prog:
            for i, batch in prog:
                if i == epoch_steps:
                    break
                self.batches_done = self.current_epoch * len(loader) + i
                sample_step = self.batches_done % self.opt.sample_interval == 0
                outs = self.model(batch, training=True, compute_norms=sample_step and self.compute_norms, global_step=self.batches_done)
                stats.update(outs['loss'])
                if sample_step:
                    if self.opt.get('log_visualizations', True):
                        save_image(outs['gen_imgs'][:16].data, self.vis_dir / ('%d_train_%d.jpg' % (self.current_epoch, i)), nrow=4, normalize=True)
                    postf = '[Batch %d/%d] [D loss: %f] [G loss: %f]' % (i, len(loader), outs['loss']['d_loss'], outs['loss']['g_loss'])
                    prog.set_postfix_str(postf, refresh=True)
                    if self.compute_norms:
                        for model_name, norms in self.model.norms.items():
                            self.logger.log(norms, self.batches_done, f'{model_name}_gradients_norm')
        epoch_stats = stats.average()
        # if self.current_epoch % self.opt.eval_counter_freq == 0:
        #     epoch_stats.update(self.evaluate_counterfactual(loader, phase='train'))
        self.logger.log(epoch_stats, self.current_epoch, 'train')
        self.logger.info(
            '[Finished training epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
        )
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()

        stats = AvgMeter()
        avg_pos_to_neg_ratio = 0.0
        for i, batch in tqdm(enumerate(loader), desc=f'Validation epoch {self.current_epoch}', leave=False, total=len(loader)):
            avg_pos_to_neg_ratio += batch[1].sum() / batch[1].shape[0]
            outs = self.model(batch, validation=True)
            stats.update(outs['loss'])
            if i % self.opt.sample_interval == 0 and self.opt.get('vis_gen', True):
                save_image(outs['gen_imgs'][:16].data, self.vis_dir / ('%d_val_%d.jpg' % (self.batches_done, i)), nrow=4, normalize=True)
        self.logger.info('[Average positives/negatives ratio in batch: %f]' % round(avg_pos_to_neg_ratio.item() / len(loader), 3))
        epoch_stats = stats.average()
        # if self.current_epoch % self.opt.eval_counter_freq == 0:
        #     epoch_stats.update(self.evaluate_counterfactual(loader, phase='val'))
        self.logger.log(epoch_stats, self.current_epoch, 'val')
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
        )
        return epoch_stats

    