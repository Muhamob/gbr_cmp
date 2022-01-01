import pytorch_lightning as pl
import numpy as np
import click 

import sys
import yaml
from pathlib import Path

from pytorch_lightning.callbacks import LearningRateMonitor

sys.path.append("../../")
sys.path.append("../../../modules/yolov5/")
from reef.yolo.callbacks import AverageEpochMetricsCallback, ProgressCallback
from utils.autoanchor import check_anchors


@click.command()
@click.option("--data-path", help="Path to dataset split directory", type=click.Path())
@click.option("--max-epochs", help="Maximum number of epochs to run", type=int)
@click.option("--num-processes", help="Number of processes to run. If greater than 1, then ddp mode is used", type=int)
@click.option("--warmup-epochs", help="Number of epochs to warmup", type=int)
@click.option("--weights", help="Path to pretrained weights", type=click.Path())
def main(data_path: str, max_epochs: int, num_processes: int, warmup_epochs: int, weights: str):
    SLICE_HEIGHT = 360
    SLICE_WIDTH = 640
    IMG_SIZE = 640
    STRIDE = 32
    RANDOM_STATE = 42
    OVERLAP_THRESHOLD = 0.25

    dataset_params = {
        'slice_height': SLICE_HEIGHT,
        'slice_width': SLICE_WIDTH,
        'overlap_threshold': OVERLAP_THRESHOLD
    }

    split_path = Path(data_path)

    with open(split_path / "data.yaml") as f:
        data = yaml.safe_load(f)

    with open("./hyp.yaml") as f:
        hyp = yaml.safe_load(f)

    from reef.yolo.lightning import YOLOModel

    module = (
        YOLOModel
        .load_from_pretrained(
            weights,
            hyp=hyp,
            data=data,
            random_state=RANDOM_STATE,
            img_size=IMG_SIZE,
            stride=STRIDE,
            dataset_params=dataset_params
        )
        .load_train_params(batch_size=4, nbs=64, warmup_epochs=warmup_epochs)
        .configure_loss()
    )
    module.setup("fit")
    module.setup("validate")

    average_epoch_metrics_callback = AverageEpochMetricsCallback()
    progress_callback = ProgressCallback()
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    print("Anchors before check:", module.model.model[-1].anchors)
    dataset = module.train_dataset
    dataset.shapes = np.array([[640, 360]] * len(module.train_dataset))
    check_anchors(module.train_dataset, module.model, imgsz=640)
    print("Anchors after check:", module.model.model[-1].anchors)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_time={"hours": 10},
        check_val_every_n_epoch=2,
        num_processes=num_processes,
        accumulate_grad_batches=module.accumulate,
        callbacks=[
            progress_callback,
            average_epoch_metrics_callback,
            learning_rate_monitor
        ]
    )
    trainer.fit(module)

    print(learning_rate_monitor.lrs)


if __name__ == "__main__":
    main()
