from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.confusion_matrix import ConfusionMatrix


class MyModel(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        compile_model: bool,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compile_model = compile_model

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging roc-auc across batches
        self.train_roc = AUROC(task="multiclass", num_classes=self.net.output_size)
        self.val_roc = AUROC(task="multiclass", num_classes=self.net.output_size)
        self.test_roc = AUROC(task="multiclass", num_classes=self.net.output_size)

        # metric objects for calculating and averaging roc-auc across batches
        self.train_confmat = ConfusionMatrix(task="multiclass", num_classes=self.net.output_size)
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=self.net.output_size)
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=self.net.output_size)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_roc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_roc.reset()
        self.val_confmat.reset()
        self.val_roc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, y)
        preds = torch.argmax(probs, dim=1)
        return loss, probs, preds, torch.argmax(y, dim=1) if y.shape[1] > 1 else y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_roc(probs, targets)
        self.train_confmat(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/roc", self.train_roc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        if self.logger is not None:
            confusion_matrix_computed = self.train_confmat.compute().detach().cpu().numpy().astype(int)

            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize=(10, 7))
            fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
            plt.close(fig_)

            self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_roc(probs, targets)
        self.val_confmat(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/roc", self.val_roc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        roc = self.val_roc.compute()  # get current val acc
        self.val_roc_best(roc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/roc_best", self.val_roc_best.compute(), sync_dist=True, prog_bar=True)

        if self.logger is not None:
            confusion_matrix_computed = self.val_confmat.compute().detach().cpu().numpy().astype(int)

            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize=(10, 7))
            fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
            plt.close(fig_)

            self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_roc(probs, targets)
        self.test_confmat(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/roc", self.test_roc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        if self.logger is not None:
            confusion_matrix_computed = self.test_confmat.compute().detach().cpu().numpy().astype(int)

            df_cm = pd.DataFrame(confusion_matrix_computed)
            plt.figure(figsize=(10, 7))
            fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
            plt.close(fig_)

            self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.


        Parameters
        ----------
        stage : str
            One of 'fit', 'validate', 'test', 'predict'. To distinguish between training,

        Returns
        -------
        None
        """
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples
        --------
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/roc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}
