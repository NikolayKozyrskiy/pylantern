from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from matches.loop import Loop
from matches.utils import single_process_only
from torch.utils.data import DataLoader

from pylantern.output_dispatcher import BaseOutputDispatcher, filter_and_uncollate
from pylantern.pipeline import BasePipeline


@single_process_only()
def predict_dataloader(
    loop: Loop,
    pipeline: BasePipeline,
    dataloader: DataLoader,
    output_dispatcher: BaseOutputDispatcher,
    group_losses: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[Any], List[Any]]:
    if save_dir is None:
        save_dir = loop.logdir / "default_infer"

    save_dir.mkdir(parents=True, exist_ok=True)

    losses, metrics = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in loop.iterate_dataloader(dataloader):
            with pipeline.batch_scope(batch):
                for f in pipeline.config.output_config:
                    f(
                        pipeline,
                        pool,
                        save_dir,
                    )
                losses_computed = {}
                if group_losses is not None:
                    for group in group_losses:
                        losses_computed.update(
                            output_dispatcher.compute_losses_group(
                                group, pipeline, loop
                            ).computed_values
                        )
                else:
                    losses_computed = output_dispatcher.compute_losses(
                        pipeline, loop
                    ).computed_values

                losses.extend(
                    filter_and_uncollate(
                        losses_computed,
                        pipeline,
                    )
                )
                metrics.extend(
                    filter_and_uncollate(
                        output_dispatcher.compute_metrics(pipeline, loop).computed_values,
                        pipeline,
                    )
                )

    def _dump(data: List, name: str):
        data = pd.DataFrame(data)
        data.to_csv(save_dir / f"{name}.csv")
        mean = data.mean(numeric_only=True)
        if verbose:
            print(f"Summary for {name}:\n{mean}")
        (save_dir / f"{name}_mean.txt").write_text(str(mean))
        return None

    _dump(losses, "losses")
    _dump(metrics, "metrics")

    return losses, metrics
