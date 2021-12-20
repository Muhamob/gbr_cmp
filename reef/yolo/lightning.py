import logging
from collections import defaultdict
from typing import Union, IO, Optional, Any, Callable

import torch
from pytorch_lightning import LightningModule
import numpy as np
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.yolo import Model
from utils.general import intersect_dicts

# configure logging at the root level of lightning
from utils.loss import ComputeLoss
from reef.yolo.data.dataset import LoadImagesAndLabels

LOGGER = logging.getLogger("pytorch_lightning")
LOGGER.setLevel(logging.DEBUG)


class YOLOModel(LightningModule):
    def __init__(
            self,
            img_size: int,
            stride: int,
            hyp: dict,
            model_yaml: dict,
            data: dict = None,
            dataset_params: dict = None,
            freeze: int = 0,
            random_state: int = None,
    ):
        super().__init__()
        self.data = data
        self.dataset_params = dataset_params
        self.img_size = img_size
        self.stride = stride

        self.random_state = random_state
        self.rs = np.random.RandomState(self.random_state)

        # model
        self.model_yaml = model_yaml
        self.hyp = hyp
        self.nc = data['nc']
        self.freeze_layers = freeze
        self.model = Model(self.model_yaml, ch=3, nc=self.nc, anchors=self.hyp.get('anchors'))
        self.configure_model()

        self.train_dataset, self.test_dataset = None, None

        self.batch_size = 4
        self.accumulate = None
        self.nbs = 64
        self.adam = False  #True

        self.compute_loss = None
        self.augment_test = False

    def configure_model(self):
        nl = self.model.model[-1].nl
        self.hyp['box'] *= 3 / nl  # scale to layers
        self.hyp['cls'] *= self.data['nc'] / 80 * 3 / nl  # scale to classes and layers
        self.hyp['obj'] *= (self.img_size / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.hyp['label_smoothing'] = False
        self.model.nc = self.data['nc']
        self.model.hyp = self.hyp
        self.model.names = self.data['names']
        return self

    def load_train_params(self, **params: dict):
        self.batch_size = params.get("batch_size") or self.batch_size
        self.nbs = params.get("nbs") or self.nbs
        self.accumulate = max(round(self.nbs / self.batch_size), 1)
        self.adam = params.get("adam") or self.adam

        self.model.hyp['weight_decay'] *= self.batch_size * self.accumulate / self.nbs

        return self

    def forward(self, imgs, **call_params) -> Any:
        imgs = imgs.float() / 255.0
        return self.model(imgs, **call_params)

    def predict_step(
            self,
            batch: Any,
            batch_idx: int,
            dataloader_idx: Optional[int] = None
    ):
        imgs = batch
        imgs = imgs.float() / 255.0
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        imgs, targets, paths, boxes = batch
        preds = self.forward(imgs)
        loss, loss_items = self.compute_loss(preds, targets)

        self.log("train_loss", loss)

        return {
            'loss': loss,
            'targets_count': list(targets.shape)[0]
        }

    def test_step(self, batch, batch_idx):
        imgs, targets, paths, boxes = batch
        preds = self.forward(imgs)
        loss, loss_items = self.compute_loss(preds[1], targets)

        self.log("test_loss", loss)

        return {
            'loss': loss,
            'targets_count': list(targets.shape)[0]
        }

    def validation_step(self, batch, batch_idx):
        imgs, targets, paths, boxes = batch
        preds = self.forward(imgs)
        loss, loss_items = self.compute_loss(preds[1], targets)

        self.log("val_loss", loss)

        return {
            'loss': loss,
            'targets_count': list(targets.shape)[0]
        }

    def configure_loss(self):
        self.compute_loss = ComputeLoss(self.model)
        return self

    def triangular_fn(self, mid_epoch: int):
        def f(epoch):
            if epoch <= mid_epoch:
                return 1.0 * epoch / mid_epoch
            else:
                return (1 - epoch / (self.trainer.max_epochs - 1)) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']

        return f

    def configure_optimizers(self):
        optimizer_param_groups = defaultdict(list)

        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                optimizer_param_groups['bias'].append(v.bias)  # g2
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                optimizer_param_groups['bn'].append(v.weight)  # g0
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                optimizer_param_groups['weights'].append(v.weight)  # g1

        if self.adam:
            optimizer = Adam(
                optimizer_param_groups["weights"],
                lr=self.hyp['lr0'],
                betas=(self.hyp['momentum'], 0.999),
                weight_decay=self.hyp['weight_decay']
            )
        else:
            optimizer = SGD(
                optimizer_param_groups["weights"],
                lr=self.hyp['lr0'],
                weight_decay=self.hyp['weight_decay'],
                momentum=self.hyp['momentum'],
                nesterov=True
            )

        optimizer.add_param_group({"params": optimizer_param_groups["bn"]})
        optimizer.add_param_group({"params": optimizer_param_groups["bias"]})

        del optimizer_param_groups

        return {
            "optimizer": optimizer,
            "lr_scheduler": LambdaLR(optimizer, self.triangular_fn(mid_epoch=3))
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            sampler=None,
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        TODO: fix test dataloader as it would be in prediction
        !!! By now it's only workaround
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            sampler=None,
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            # sampler=None,
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            print("Stage:", stage)
            self.train_dataset = LoadImagesAndLabels(
                path=self.data['train'],
                slice_height=self.dataset_params['slice_height'],
                slice_width=self.dataset_params['slice_width'],
                overlap_threshold=self.dataset_params['overlap_threshold'],
                img_size=self.img_size,
                augment=True,
                stride=self.stride,
                random_state=self.random_state,
                hyp=self.hyp,
                prefix="train: ",
            )
        elif stage == 'test' or stage == 'validate':
            self.test_dataset = LoadImagesAndLabels(
                path=self.data['val'],
                slice_height=self.dataset_params['slice_height'],
                slice_width=self.dataset_params['slice_width'],
                overlap_threshold=self.dataset_params['overlap_threshold'],
                img_size=self.img_size,
                augment=False,
                stride=self.stride,
                random_state=self.random_state,
                hyp=self.hyp,
                prefix="test: ",
            )
        else:
            raise Exception(f"Stage must be one of train/test; given {stage}")

    def freeze(self, layers: int = 0):
        freeze = [f'model.{x}.' for x in range(layers)]  # layers to freeze

        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                LOGGER.info(f'freezing {k}')
                v.requires_grad = False
        LOGGER.info(f'Number of freezed layers: {len(freeze)}')

        return self

    def unfreeze(self):
        for k, v in self.model.named_parameters():
            v.requires_grad = True
        return self

    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint_path: Union[str, IO],
        **kwargs,
    ):
        def get_param(ckpt: dict, name: str, default: Any = None, nullable: bool = False):
            result = kwargs.get(name) or ckpt.get(name) or default
            if result is None and not nullable:
                raise Exception(f"{name} is null")
            else:
                return result

        ckpt = torch.load(checkpoint_path)
        hyp = get_param(ckpt, "hyp")
        img_size = get_param(ckpt, "img_size")
        stride = get_param(ckpt, "stride")
        model_yaml = get_param(ckpt, "model_yaml", default=ckpt['model'].yaml)
        data = get_param(ckpt, "data")
        dataset_params = get_param(ckpt, "dataset_params")
        freeze = get_param(ckpt, "freeze", default=0)
        random_state = get_param(ckpt, "random_state", default=42)

        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3

        module = cls(
            img_size=img_size,
            stride=stride,
            hyp=hyp,
            model_yaml=model_yaml,
            data=data,
            dataset_params=dataset_params,
            freeze=freeze,
            random_state=random_state
        )

        # exclude params
        exclude = ['anchor'] if hyp.get('anchors') else []

        # load model
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, module.model.state_dict(), exclude=exclude)
        module.model.load_state_dict(csd, strict=False)
        LOGGER.debug(f"Loaded model from checkpoint {checkpoint_path}")

        return module
