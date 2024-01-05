import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .utils import mkdir


def make_symlinks(
    src_dir: Path,
    dst_dir: Path,
    ext: str,
    exclude: Optional[Path] = None,
    include: Optional[Path] = None,
):
    mkdir(dst_dir)
    names = {p.stem for p in src_dir.glob(f"*.{ext}")}

    if exclude is not None:
        names.difference_update(json.loads(exclude.read_text()))
    if include is not None:
        names.intersection_update(json.loads(include.read_text()))
    for name in tqdm(sorted(list(names))):
        (dst_dir / f"{name}.{ext}").symlink_to(src_dir / f"{name}.{ext}")

    return None


def make_split(
    data_root: Path,
    output_dir: Path,
    folds: int = 10,
    exclude: Optional[Path] = None,
    include: Optional[Path] = None,
    split_name: str = "default",
    folds_prefix: str = "",
):
    if len(folds_prefix):
        folds_prefix = folds_prefix + "_"

    seq_names = {p.stem for p in data_root.glob("*")}

    if exclude is not None:
        seq_names.difference_update(json.loads(exclude.read_text()))
    if include is not None:
        seq_names.intersection_update(json.loads(include.read_text()))

    random.seed(22)
    print(f"Total {len(seq_names)} found")

    color_by_seq = defaultdict(list)
    for s in seq_names:
        color_by_seq[s].append(s)

    seq_names = [*seq_names]
    random.shuffle(seq_names)
    splits = defaultdict(list)
    for i, seq in enumerate(seq_names):
        splits[i % folds] += color_by_seq[seq]

    split_path = mkdir(output_dir / f"splits/{split_name}")
    for k, v in splits.items():
        (split_path / f"{folds_prefix}{k}.json").write_text(json.dumps(v, indent=2))

    return None


def read_split(split_path: Path, fold_ids: List[int]) -> List[str]:
    result = []
    if len(fold_ids) == 0:
        fold_ids = [p.stem for p in split_path.glob("*.json")]

    for idx in fold_ids:
        with (split_path / f"{idx}.json").open() as f:
            result += json.load(f)

    return result
