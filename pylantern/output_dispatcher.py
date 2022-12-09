from typing import Dict, NamedTuple, List

import torch
from torch import Tensor

from .pipeline import Pipeline


class LossDispatchResult(NamedTuple):
    computed_values: Dict[str, Tensor]
    aggregated: Tensor


class MetricsDispatchResult(NamedTuple):
    computed_values: Dict[str, Tensor]


class OutputDispatcher:
    def __init__(
        self,
        loss_aggregation_weigths: Dict[str, float],
        metrics: List[str],
    ):
        self.loss_aggregation_weigths = loss_aggregation_weigths
        self.metrics = metrics

        self._prepare_loss_fns()
        self._prepare_metric_fns()

    def compute_losses(self, pipeline: Pipeline) -> LossDispatchResult:
        loss_values = {
            name.replace("__", "/"): getattr(self, name)(pipeline)
            for name in self.train_loss_fn
        }

        with torch.no_grad():
            loss_values.update(
                {
                    fn_name.replace("__", "/"): getattr(self, fn_name)(pipeline)
                    for fn_name in self.eval_loss_fn
                }
            )

        for k in list(loss_values.keys()):
            if isinstance(loss_values[k], dict):
                d = loss_values.pop(k)
                loss_values.update({f"{k}.{in_k}": v for in_k, v in d.items()})

        losses, weights = [], []
        for name, w in self.loss_aggregation_weigths.items():
            losses.append(loss_values[name].mean())
            weights.append(w)
        if len(losses) > 0:
            losses = torch.stack(losses)
            weights = losses.new_tensor(weights)

            aggregated = (losses * weights).sum() / weights.sum()
        else:
            aggregated = torch.tensor(0)

        loss_values["_total"] = aggregated
        return LossDispatchResult(loss_values, aggregated)

    @torch.no_grad()
    def compute_metrics(self, pipeline: Pipeline) -> MetricsDispatchResult:
        metric_values = {
            name.replace("__", "/"): getattr(self, name)(pipeline)
            for name in self.metric_fn
        }

        return MetricsDispatchResult(metric_values)

    def _prepare_metric_fns(self):
        self.metric_fn = list(
            {n.replace("/", "__").split(".")[0] for n in self.metrics}
        )

    def _prepare_loss_fns(self):
        self.train_loss_fn = list(
            {
                n.replace("/", "__").split(".")[0]
                for n, weight in self.loss_aggregation_weigths.items()
                if weight != 0
            }
        )

        self.eval_loss_fn = list(
            {
                n.replace("/", "__").split(".")[0]
                for n, weight in self.loss_aggregation_weigths.items()
                if weight == 0
            }
        )
        self.eval_loss_fn = [
            n for n in self.eval_loss_fn if n not in self.train_loss_fn
        ]
        self.loss_aggregation_weigths = {
            name: w for name, w in self.loss_aggregation_weigths.items() if w != 0
        }
        return None


def filter_and_uncollate(batch_values: Dict[str, Tensor]):
    batch_values = {k: v.tolist() for k, v in batch_values.items() if k != "_total"}
    return uncollate(batch_values)


def uncollate(params: Dict):
    params = [
        dict(zip(params.keys(), t)) for t in zip(*[params[k] for k in params.keys()])
    ]
    return params
