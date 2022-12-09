from typing import List

from matches.callbacks import Callback
from matches.loop import Loop
from matches.loop.metric_manager import MetricEntry
from pydantic import BaseModel
from wandb.sdk.wandb_run import Run

import wandb
from wandb import Settings


class WandBLoggingSink(Callback):
    def __init__(self, comment: str, config: BaseModel):
        self.config = config
        self.comment = comment
        self.model_watched = False
        self.run = None
        self.loader_mode = None

    def on_train_start(self, loop: "Loop"):
        self.loader_mode = loop._loader_override.mode
        if self.loader_mode == "disabled":
            self.loader_mode = "full"

        config_dict = self.config.dict()
        config_dict["logdir"] = str(loop.logdir)
        self.run: Run = wandb.init(
            dir=str(loop.logdir),
            name=self.comment,
            tags=[self.loader_mode],
            settings=Settings(silent=True, console="off"),
            config=config_dict,
        )

    def on_train_end(self, loop: "Loop"):
        self._consume_new_entries(loop)
        self.run.finish()
        print("Finish!")

    def on_iteration_end(self, loop: "Loop", batch_no: int):
        self._consume_new_entries(loop)

    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        self._consume_new_entries(loop)

    def _consume_new_entries(self, loop):
        entries: List[MetricEntry] = loop.metrics.collect_new_entries(reset=False)
        for e in entries:
            entry = {
                e.name: e.value,
                **{t.value: v for t, v in e.iteration_values.items()},
            }
            self.run.log(entry, commit=False)
        self.run.log({}, commit=True)

    def on_epoch_start(self, loop: "Loop", epoch_no: int, total_epochs: int):
        if not self.model_watched:
            self.model_watched = True
            wandb.watch(
                loop._modules,
                log="all",
                log_freq=10 if self.loader_mode != "full" else 100,
            )
