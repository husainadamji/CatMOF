"""Task presets: which target column and default filenames to use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BpnnTask:
    """One regression task (oxo vs HAT): picks targets CSV + column; features come from a separate pickle."""

    name: str
    targets_csv: str
    sbu_column: str
    target_column: str


DEFAULT_TASKS: Mapping[str, BpnnTask] = {
    "oxo": BpnnTask(
        name="oxo",
        targets_csv="targets_outliers_removed.csv",
        sbu_column="sbu_name",
        target_column="DEoxo (kcal/mol)",
    ),
    "hat": BpnnTask(
        name="hat",
        targets_csv="targets_outliers_removed.csv",
        sbu_column="sbu_name",
        target_column="DEHAT (kcal/mol)",
    ),
}


def get_task(key: str) -> BpnnTask:
    if key not in DEFAULT_TASKS:
        raise KeyError(f"Unknown task {key!r}; choose one of {sorted(DEFAULT_TASKS)}")
    return DEFAULT_TASKS[key]
