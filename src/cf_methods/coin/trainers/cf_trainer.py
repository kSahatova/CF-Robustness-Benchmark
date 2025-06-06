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
# from src.cf_methods.coin.utils import confmat_vis_img
from src.utils import evaluate_classification_model



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
        # self.val_iou_xc = BinaryJaccardIndex(self.cf_threshold).to(self.device)
        # self.val_iou_xfx = BinaryJaccardIndex(self.cf_threshold).to(self.device)

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
        if self.current_epoch % self.opt.eval_counter_freq == 0:
            epoch_stats.update(self.evaluate_counterfactual(loader, phase='train'))
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
        if self.current_epoch % self.opt.eval_counter_freq == 0:
            epoch_stats.update(self.evaluate_counterfactual(loader, phase='val'))
        self.logger.log(epoch_stats, self.current_epoch, 'val')
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
        )
        return epoch_stats

    @torch.no_grad()
    def evaluate_counterfactual(self, loader, phase='val', tau=0.8, skip_fid=False, postprocess_morph=False):
        self.model.eval()

        cf_dir = self.cf_vis_dir_train if phase == 'train' else self.cf_vis_dir_val
        classes = []
        cv_y_true, cv_y_pred = [], []
        posterior_true, posterior_pred = [], []

        # number of samples where classifier predicted > 0.8 and the gt label is 1 (abnormal)
        pred_num_abnormal_samples = 0
        true_num_abnormal_samples = 0
        for i, batch in tqdm(enumerate(loader), desc='Validating counterfactuals', leave=False, total=len(loader)):
            # Evaluate Counterfactual Validity Metric
            real_imgs = batch[0].cuda(non_blocking=True)
            # cf_gt_masks = batch['masks'][:, self.cf_gt_seg_mask_idx].cuda(non_blocking=True)
            labels = batch[1]
            true_num_abnormal_samples += labels.sum()
            B = labels.shape[0]

            self.model: CounterfactualCGAN
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.model.posterior_prob(real_imgs)

            real_neg_pos_group = ((real_f_x < 0.2) | (real_f_x > 0.8)).view(-1)
            if not real_neg_pos_group.any():
                continue

            # filter out samples not belonging to either real negative or positive groups
            # real_imgs, cf_gt_masks, labels = real_imgs[real_neg_pos_group], cf_gt_masks[real_neg_pos_group], labels[real_neg_pos_group.cpu()]
            real_imgs, labels = real_imgs[real_neg_pos_group], labels[real_neg_pos_group.cpu()]

            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = (
                real_f_x[real_neg_pos_group],
                real_f_x_discrete[real_neg_pos_group],
                real_f_x_desired[real_neg_pos_group],
                real_f_x_desired_discrete[real_neg_pos_group],
            )

            # our ground truth is the `flipped` labels
            cv_y_true.extend(real_f_x_desired_discrete.cpu().squeeze(1).numpy())
            posterior_true.extend(real_f_x.cpu().squeeze(1).numpy())
            classes.extend(labels.cpu().numpy())

            # computes I_f(x, c)
            gen_cf_c = self.model.explanation_function(real_imgs, real_f_x_desired_discrete)
            
            # computes I_f(x, f(x))
            gen_cf_fx = self.model.explanation_function(real_imgs, real_f_x_discrete)
            # print(real_f_x_discrete.shape, gen_cf_fx.shape, real_imgs.shape)

            # computes f(x_c)
            gen_f_x, gen_f_x_discrete, _, _ = self.model.posterior_prob(gen_cf_c)
            # our prediction is the classifier's label for the generated images given the desired posterior probability
            cv_y_pred.extend(gen_f_x_discrete.cpu().squeeze(1).numpy())
            posterior_pred.extend(gen_f_x.cpu().squeeze(1).numpy())

            # denorm values from [-1; 1] to [0, 1] range, B x 1 x H x W
            real_imgs.add_(1).div_(2)
            gen_cf_c.add_(1).div_(2)
            gen_cf_fx.add_(1).div_(2)

            # compute difference maps, threshold and compute IoU
            # |x - x_c|
            diff = (real_imgs - gen_cf_c).abs() # [0; 1] values
            diff_seg = (diff > self.cf_threshold).byte()
            
            # |x_fx - x_c|
            diff2 = (gen_cf_fx - gen_cf_c).abs() # [0; 1] values
            diff2_seg = (diff2 > self.cf_threshold).byte()
            
            # abnormal_mask = labels.bool()
            # if abnormal_mask.any():
            #   pred_num_abnormal_samples += abnormal_mask.sum()
                # self.val_iou_xc.update(diff_seg[abnormal_mask].squeeze(1), cf_gt_masks[abnormal_mask].squeeze(1))
                # self.val_iou_xfx.update(diff2_seg[abnormal_mask].squeeze(1), cf_gt_masks[abnormal_mask].squeeze(1))
            
            # diff3 = (real_imgs[0] - gen_cf_fx[0]).abs() # [0; 1] values
            # vis = torch.stack((
            #     real_imgs[0], torch.zeros_like(real_imgs[0]), torch.zeros_like(real_imgs[0]), 
            #     gen_cf_c[0], diff[0], diff_seg[0],
            #     gen_cf_fx[0], diff2[0], diff2_seg[0],
            # ), dim=0).permute(0, 2, 3, 1)
            # vis = torch.cat((vis, vis, vis), 3)
            # vis_confmat = confmat_vis_img(cf_gt_masks[0].unsqueeze(0).unsqueeze(0), diff_seg[0].unsqueeze(0), normalized=True)[0]
            # vis[1] = 0.3*vis[0] + 0.7 * vis_confmat
            # vis[2] = vis_confmat
            # vis = vis.permute(0, 3, 1, 2)
            
            # # save first example for visualization
            # vis_path = cf_dir / (f'epoch_%d_counterfactual_%d_label_%d_true_%d_pred_%d.jpg' % (
            #     self.current_epoch, i, labels[0], real_f_x_desired_discrete[0][0], gen_f_x_discrete[0][0])
            # )
            # save_image(vis.data, vis_path, nrow=3, normalize=False) # value_range=(-1, 1))

            if not skip_fid:
                # Evaluate Frechet Inception Distance (FID)
                # upsample to InceptionV3's resolution and convert to RGB
                # real_imgs = nn.functional.interpolate(real_imgs, size=(299, 299), mode='bilinear', align_corners=False)
                # real_imgs = real_imgs.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(real_imgs, real=True)
                
                # upsample to InceptionV3's resolution and convert to RGB
                # gen_cf_c = nn.functional.interpolate(gen_cf_c, size=(299, 299), mode='bilinear', align_corners=False)
                # gen_cf_c = gen_cf_c.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(gen_cf_c, real=False)

        num_samples = len(posterior_true)
        self.logger.info(f'Finished evaluating counterfactual results for epoch: {self.current_epoch}')

        # Counterfactual Accuracy (flip rate) Score
        cv_y_true, cv_y_pred = np.array(cv_y_true), np.array(cv_y_pred)
        cacc = np.mean(cv_y_true == cv_y_pred)
        self.logger.info(f'Counterfactual accuracy = {cacc} (num_samples={len(cv_y_pred)})')

        posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
        # Counterfactual Validity Score
        cv_score = np.mean(np.abs(posterior_true - posterior_pred) > tau)
        self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau}, num_samples={num_samples})')

        # cf_iou_xc = self.val_iou_xc.compute().item()
        # self.val_iou_xc.reset()
        # self.logger.info(f'IoU(S, Sc) = {cf_iou_xc:.3f} (cf_thresh={self.cf_threshold}, num_samples={pred_num_abnormal_samples}, mask={self.cf_gt_seg_mask_idx})')

        # cf_iou_xfx = self.val_iou_xfx.compute().item()
        # self.val_iou_xfx.reset()
        # self.logger.info(f'IoU(S, Sfx) = {cf_iou_xfx:.3f} (cf_thresh={self.cf_threshold}, num_samples={pred_num_abnormal_samples}, mask={self.cf_gt_seg_mask_idx})')

        fid_score = None
        if not skip_fid:
            # Frechet Inception Distance (FID) Score
            fid_score = self.val_fid.compute().item()
            self.logger.info(f'FID(X, Xc) = {fid_score:.3f} (num_samples={num_samples}, features={self.fid_features})')
            self.val_fid.reset()
        
        self.logger.info(f'Ratio of true abnormal slices to classified as abnormal slices: {pred_num_abnormal_samples / (max(true_num_abnormal_samples, 1e-8))}')
        return {
            'counter_acc': cacc,
            f'cv_{int(tau*100)}': cv_score,
            'fid': fid_score,
            # 'cf_iou_xc': cf_iou_xc,
        }
    