import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
import numpy as np

import importlib

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm


class ProgressCallback(TQDMProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._train_batch_idx += 1
        if self._should_update(self.train_batch_idx):
            self._update_bar(self.main_progress_bar)

            prefix_dict = self.get_metrics(trainer, pl_module)
            prefix_dict['labels'] = outputs['targets_count']

            self.main_progress_bar.set_postfix(prefix_dict)

    # def on_before_optimizer_step(
    #         self,
    #         trainer: "pl.Trainer",
    #         pl_module: "pl.LightningModule",
    #         optimizer,
    #         opt_idx: int
    # ) -> None:
    #     self.main_progress_bar.write("On before optimizer step")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._test_batch_idx += 1
        if self._should_update(self.test_batch_idx):
            self._update_bar(self.test_progress_bar)

            prefix_dict = self.get_metrics(trainer, pl_module)
            prefix_dict['labels'] = outputs['targets_count']

            self.test_progress_bar.set_postfix(prefix_dict)


class AverageEpochMetricsCallback(Callback):
    def __init__(self):
        self.outputs = {
            'train': [],
            'val': [],
            'test': [],
        }

        self.mean_outputs = {
            'train': [],
            'val': [],
            'test': [],
        }

    def _update_outputs(self, stage: str, batch_idx: int, values: dict):
        self.outputs[stage].append({
            'batch_idx': batch_idx,
            **values
        })

    def _average_predictions(self, stage: str):
        result = dict()
        names = [k for k in self.outputs[stage][0].keys() if k != 'batch_idx']

        for name in names:
            result[name] = np.mean([output[name] for output in self.outputs[stage]])

        return result

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: dict,
            batch,
            batch_idx: int,
            unused: int = 0,
    ):
        stage = "train"
        outputs['loss'] = outputs['loss'].item()
        self._update_outputs(stage, batch_idx, outputs)

    def on_train_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ):
        stage = "train"
        avg_metrics = self._average_predictions(stage)
        self.mean_outputs[stage].append(avg_metrics)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: dict,
            batch,
            batch_idx: int,
            unused: int = 0,
    ):
        stage = "val"
        outputs['loss'] = outputs['loss'].item()
        self._update_outputs(stage, batch_idx, outputs)

    def on_validation_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ):
        stage = "val"
        avg_metrics = self._average_predictions(stage)
        self.mean_outputs[stage].append(avg_metrics)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: dict,
            batch,
            batch_idx: int,
            unused: int = 0,
    ):
        stage = "test"
        outputs['loss'] = outputs['loss'].item()
        self._update_outputs(stage, batch_idx, outputs)

    def on_test_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ):
        stage = "test"
        avg_metrics = self._average_predictions(stage)
        self.mean_outputs[stage].append(avg_metrics)
