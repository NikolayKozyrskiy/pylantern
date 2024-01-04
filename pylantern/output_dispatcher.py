from collections import OrderedDict
from types import ModuleType
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from ignite.metrics import Metric
from matches.loop import Loop
from torch import Tensor

from pylantern.common.utils import attrdict

if TYPE_CHECKING:
    from pylantern.config import BaseConfig
    from pylantern.pipeline import BasePipeline


class LossDispatchResult(NamedTuple):
    computed_values: Dict[str, Tensor]
    aggregated: Tensor
    computed_values_weighted: Optional[Dict[str, Tensor]] = None


class MetricsDispatchResult(NamedTuple):
    computed_values: Dict[str, Tensor]


class CriterionGroup(NamedTuple):
    loss_aggregation_weigths: OrderedDict[str, float]
    loss_fn_names: List[str]


class CriterionAggregation:
    def __init__(self, default: Optional[Dict[str, float]] = None, **kwargs) -> None:
        self._criterion_groups: Dict[str, CriterionGroup] = {}
        self._unique_loss_fn_names: List[str] = []
        self._unique_loss_aggregation_weigths: OrderedDict[str, float] = OrderedDict()

        kwargs["default"] = default if default is not None else {}
        self._init_criterion_groups(**kwargs)

    @property
    def unique_loss_fn_names(self) -> List[str]:
        return self._unique_loss_fn_names

    @property
    def unique_loss_aggregation_weigths(self) -> OrderedDict[str, float]:
        return self._unique_loss_aggregation_weigths

    def get_group(self, group_name: str) -> Optional[CriterionGroup]:
        return self._criterion_groups.get(group_name, None)

    def has_group(self, group_name: str) -> bool:
        return self.get_group(group_name) is not None

    def _init_criterion_groups(self, **kwargs) -> None:
        for group_name, group in kwargs.items():
            group: Dict[str, float]
            loss_aggregation_weigths = OrderedDict(group)
            loss_fn_names = sorted(
                [n.replace("/", "__").split(".")[0] for n, weight in group.items()]
            )
            self._criterion_groups[group_name] = CriterionGroup(
                loss_aggregation_weigths=loss_aggregation_weigths,
                loss_fn_names=loss_fn_names,
            )
            self._unique_loss_fn_names += loss_fn_names
            self._unique_loss_aggregation_weigths.update(group)

        self._unique_loss_fn_names = sorted(list(set(self._unique_loss_fn_names)))
        return None


class BaseOutputDispatcher:
    def __init__(
        self,
        config: "BaseConfig",
        complex_criterions_module: Optional[ModuleType] = None,
        *args,
        **kwargs,
    ):
        self.criterion_aggregation = config.criterion_aggregation
        self.normalize_losses = config.normalize_losses

        self.metric_fn_names: List[str] = None
        self.complex_criterions = attrdict()

        self._prepare_metric_fns(config.metrics)
        if complex_criterions_module is not None:
            assert hasattr(
                complex_criterions_module, "LOSS_CRITERION_MAP"
            ), "complex_criterions_module must have LOSS_CRITERION_MAP"
            self._prepare_complex_criterions(
                complex_criterions_module=complex_criterions_module, *args, **kwargs
            )

    def has_group(self, group_name: str) -> bool:
        return self.criterion_aggregation.has_group(group_name)

    def compute_losses_group(
        self,
        group_name: str,
        pipeline: "BasePipeline",
        loop: "Loop",
        losses_avg_dict: Optional[Dict[str, Metric]] = None,
        *args,
        **kwargs,
    ) -> LossDispatchResult:
        group = self.criterion_aggregation.get_group(group_name)
        return self._compute_losses(
            loss_fn_names=group.loss_fn_names,
            loss_aggregation_weigths=group.loss_aggregation_weigths,
            pipeline=pipeline,
            loop=loop,
            losses_avg_dict=losses_avg_dict,
            *args,
            **kwargs,
        )

    def compute_losses(
        self,
        pipeline: "BasePipeline",
        loop: "Loop",
        losses_avg_dict: Optional[Dict[str, Metric]] = None,
        *args,
        **kwargs,
    ) -> LossDispatchResult:
        return self._compute_losses(
            loss_fn_names=self.criterion_aggregation.unique_loss_fn_names,
            loss_aggregation_weigths=self.criterion_aggregation.unique_loss_aggregation_weigths,
            pipeline=pipeline,
            loop=loop,
            losses_avg_dict=losses_avg_dict,
            *args,
            **kwargs,
        )

    def _compute_losses(
        self,
        loss_fn_names: List[str],
        loss_aggregation_weigths: OrderedDict[str, float],
        pipeline: "BasePipeline",
        loop: "Loop",
        losses_avg_dict: Optional[Dict[str, Metric]] = None,
        *args,
        **kwargs,
    ) -> LossDispatchResult:
        loss_values = {
            name: getattr(self, name)(
                pipeline,
                loop,
                *args,
                **kwargs,
            )
            for name in loss_fn_names
        }

        for k in list(loss_values.keys()):
            if isinstance(loss_values[k], dict):
                d = loss_values.pop(k)
                loss_values.update({f"{k}.{in_k}": v for in_k, v in d.items()})

        losses, weights = [], []
        for name, w in loss_aggregation_weigths.items():
            _loss_value = loss_values[name.replace("/", "__")]
            if self.normalize_losses:
                _loss_value = _loss_value / _loss_value.item()
            losses.append(_loss_value)
            weights.append(w)
        if len(losses) > 0:
            losses = torch.stack(losses)
            weights = losses.new_tensor(weights)
            aggregated = (losses * weights).sum()
        else:
            aggregated = torch.tensor(0)

        loss_values["_total"] = aggregated

        if losses_avg_dict is not None:
            with torch.no_grad():
                for name in loss_fn_names:
                    losses_avg_dict[name.replace("__", "/")].update(
                        loss_values[name].detach()
                    )

        return LossDispatchResult(computed_values=loss_values, aggregated=aggregated)

    @torch.no_grad()
    def compute_metrics(
        self,
        pipeline: "BasePipeline",
        loop: "Loop",
        metrics_avg_dict: Optional[Dict[str, Metric]] = None,
        *args,
        **kwargs,
    ) -> Union[MetricsDispatchResult, Tuple[MetricsDispatchResult, Dict]]:
        metric_values = {
            name: getattr(self, name)(
                pipeline,
                loop,
                *args,
                **kwargs,
            )
            for name in self.metric_fn_names
        }

        if metrics_avg_dict is not None:
            for name in self.metric_fn_names:
                metrics_avg_dict[name.replace("__", "/")].update(
                    metric_values[name].detach().mean()
                )

        return MetricsDispatchResult(metric_values)

    def _prepare_metric_fns(self, metrics: List[str]):
        self.metric_fn_names = sorted(
            [n.replace("/", "__").split(".")[0] for n in metrics]
        )
        return None

    def _prepare_complex_criterions(
        self, complex_criterions_module: ModuleType, *args, **kwargs
    ) -> None:
        for loss_name in self.criterion_aggregation.unique_loss_fn_names:
            if loss_name not in complex_criterions_module.LOSS_CRITERION_MAP.keys():
                continue
            criterions = complex_criterions_module.LOSS_CRITERION_MAP[loss_name]
            for criterion in criterions:
                if criterion not in self.complex_criterions.keys():
                    self.complex_criterions[criterion] = getattr(
                        complex_criterions_module, criterion
                    )(*args, **kwargs).to(kwargs.get("device"))

        return None


def filter_and_uncollate(batch_values: Dict[str, Tensor], pipeline: "BasePipeline"):
    batch_values = {
        k: v[None].tolist() for k, v in batch_values.items() if k != "_total"
    }
    return uncollate(batch_values)


def uncollate(params: Dict):
    params = [
        dict(zip(params.keys(), t)) for t in zip(*[params[k] for k in params.keys()])
    ]
    return params
