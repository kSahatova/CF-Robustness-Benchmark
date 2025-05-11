# pyright: reportMissingImports=false
import os
import torch
# from torchinfo import summary
from matplotlib import pyplot as plt
from src.models import vae 
from lightning.pytorch import LightningModule


class VAELightningWrapper(LightningModule):
    def __init__(
        self,
        config,
        model,
    ):
        """
        Initialize the LightningCNN module for a given classifier.
        """
        super().__init__()
        # saves all the hyperparameters in the hparams.yaml file
        self.model = model
        loss_class = getattr(torch.nn, config.loss.name)
        self.loss = loss_class(**config.loss.args)

        self.optimizer = getattr(torch.optim, config.optimizer.name)
        self.optim_args = config.optimizer.args

        self.kld_annealer = getattr(vae, config.annealer.name)
        self.kld_annealer = self.kld_annealer(**config.annealer.args)

        self.num_classes = config.data.num_classes
        self.task = "binary" if self.num_classes == 2 else "multiclass"



    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, labels = batch
        recon_images, mu, logvar = self.model(images)
        recon = self.loss(recon_images, images)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.kld_annealer is not None:
            KLD = self.kld_annealer(KLD)

        total_loss = recon + self.model.beta * KLD

        self.log("train_total", total_loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("train_recon", recon, prog_bar=True, logger=True, on_epoch=True)
        self.log("train_kld", KLD, prog_bar=True, logger=True, on_epoch=True)
        # adds a scalar that will combine on the plot both training and validation loss values
        # self.logger.experiment.add_scalars("loss", {"train": loss}, self.global_step)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            images, labels = batch
            z, _ = self.model.encoder(images)
            recon_images = self.model.decoder(z)
            recon_images = recon_images.to(self.device)

            # plot reconstructed images
            fig, axes = plt.subplots(2, 8, figsize=(15, 4))
            for i in range(8):
                axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().squeeze(), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(recon_images[i].permute(1, 2, 0).cpu().squeeze(), cmap='gray')
                axes[1, i].axis('off')
            # fig.savefig(os.path.join(self.vis_dir, f'reconstructed_imgs_{self.current_epoch}.png'))
            plt.show()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optim_args)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k.partition('model.')[2]: v for k,v in checkpoint['state_dict'].items()}
