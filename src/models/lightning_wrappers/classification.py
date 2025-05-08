# pyright: reportMissingImports=false

import torch
import torchmetrics
# from torchinfo import summary

from lightning.pytorch import LightningModule


class ClassifierLightningWrapper(LightningModule):
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
        self.loss_fn = loss_class(**config.loss.args)

        self.optimizer = getattr(torch.optim, config.optimizer.name)
        self.optim_args = config.optimizer.args

        self.num_classes = config.data.num_classes
        self.task = "binary" if self.num_classes == 2 else "multiclass"

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.classification.Accuracy(
                    task=self.task, num_classes=self.num_classes
                ),
                "auc": torchmetrics.AUROC(task=self.task, num_classes=self.num_classes),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="val_")
        self.save_hyperparameters()
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, labels = batch
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        # adds a scalar that will combine on the plot both training and validation loss values
        # self.logger.experiment.add_scalars("loss", {"train": loss}, self.global_step)
        if self.task == "binary":
            predictions = torch.argmax(logits, dim=1)
        else:
            predictions = logits
        self.train_metrics.update(predictions, labels)

        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        images, labels = batch
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        # self.logger.experiment.add_scalars("loss", {"val": loss}, self.global_step)
        if self.task == "binary":
            predictions = torch.argmax(logits, dim=1)
        else:
            predictions = logits
        self.valid_metrics.update(predictions, labels)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optim_args)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k.partition('model.')[2]: v for k,v in checkpoint['state_dict'].items()}
