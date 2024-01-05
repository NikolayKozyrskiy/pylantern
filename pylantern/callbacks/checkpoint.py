import logging

from ignite.distributed import one_rank_only
from matches.callbacks import Callback
from matches.loop import Loop

LOG = logging.getLogger(__name__)


class EveryEpochModelSaver(Callback):
    def __init__(
        self, model_name: str, save_each_epochs: int = 1, logdir_suffix: str = ""
    ):
        self.model_name = model_name
        self.save_each_epochs = save_each_epochs
        self.logdir_suffix = logdir_suffix

    @one_rank_only()
    def on_epoch_end(self, loop: "Loop", epoch_no: int, total_epochs: int):
        self._save_state_dict(loop, epoch_no)

    def _save_state_dict(self, loop: "Loop", epoch_no: int):
        if (epoch_no + 1) % self.save_each_epochs == 0:
            checkpoint_path = (
                loop.logdir / self.logdir_suffix / f"{self.model_name}_{epoch_no}.pth"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            loop.state_manager.write_state_by_key(checkpoint_path, key=self.model_name)
