from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from flopco import FlopCo
from torch import Tensor

from pylantern.common.utils import dump_txt, get_device, mkdir


@torch.no_grad()
def module_speed_test(
    logdir: Path,
    model: nn.Module,
    inputs: Union[Tensor, List[Tensor], Dict[str, Tensor]],
    input_sizes: Union[List[int], List[List[int]]],
    repeats: int = 10000,
    postfix: str = "",
) -> None:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(100):
        model(inputs)
    torch.cuda.synchronize()
    for _ in range(repeats):
        starter.record()
        model(inputs)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

    timings = np.array(sorted(timings)) / 1000
    res_str = (
        f"Inference speed test for input size={input_sizes} result in sec:"
        + f"\n\tmean +- 3*std: {timings.mean()} +- {3 * timings.std()}"
        + f"\n\ttop 5 min: {timings[:5]}"
        + f"\n\ttop 5 max: {timings[-5:]}"
        + f"\n\tFPS: {1 / timings.mean()}"
    )
    print(res_str)
    mkdir(logdir)
    dst = f"{logdir}/inference_speed_{postfix}__{input_sizes}.txt"
    dump_txt(res_str, dst)
    return None


@torch.no_grad()
def jit_speed_test(
    logdir: Path,
    inputs: Union[Tensor, List[Tensor], Dict[str, Tensor]],
    input_sizes: Union[List[int], List[List[int]]],
    checkpoint_name: str = "best",
    repeats: int = 10000,
    postfix: str = "",
) -> None:
    device = get_device()
    model_path = f"{logdir}/{checkpoint_name}.pt"
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(100):
        model.forward(inputs)
    torch.cuda.synchronize()
    for _ in range(repeats):
        starter.record()
        model.forward(inputs)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

    timings = np.array(sorted(timings)) / 1000
    res_str = (
        f"Inference speed test for input size={input_sizes} result in sec:"
        + f"\n\tmean +- 3*std: {timings.mean()} +- {3 * timings.std()}"
        + f"\n\ttop 5 min: {timings[:5]}"
        + f"\n\ttop 5 max: {timings[-5:]}"
        + f"\n\tFPS: {1 / timings.mean()}"
    )
    print(res_str)
    dst = f"{logdir}/inference_speed_jit_{postfix}__{input_sizes}.txt"
    dump_txt(res_str, dst)
    return None


def model_stats(
    logdir: Path,
    model: nn.Module,
    inputs: Dict[str, Tensor],
    input_sizes: Union[List[int], List[List[int]]],
    postfix: str = "",
) -> None:
    device = get_device()
    model = model.eval()

    instances = [
        nn.Conv2d,
        nn.Linear,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.Softmax,
    ]
    model_stats = FlopCo(
        model, custom_tensor=inputs, instances=instances, device=device
    )

    res_str = f"Total number of params: {model_stats.total_params:,}\n"
    res_str += (
        f"Total number of FLOPs: {model_stats.total_flops:,} "
        + f"for input size={input_sizes}"
    )
    print(res_str)
    mkdir(logdir)
    dst = f"{logdir}/model_stats_{postfix}__{input_sizes}.txt"
    dump_txt(res_str, dst)
    return None
