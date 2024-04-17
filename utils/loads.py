import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, TextIO, Type, TypeVar, Union

from utils.types import Annotation, Graph, Plan, TrajectoryData

T = TypeVar("T")


def json_loadlines(f: TextIO, astype: Type[T] = Dict[str, Any]) -> Iterator[T]:
    while line := f.readline():
        yield json.loads(line)


def load_plan(filename: Union[str, Path]) -> Plan:
    with open(filename, "r") as f:
        return json.load(f)


def load_environment_states(filename: Union[str, Path]) -> list[Graph]:
    with open(filename, "r") as f:
        return list(json_loadlines(f, astype=Graph))


def load_annotations(filename: Union[str, Path]) -> list[Annotation]:
    with open(filename, "r") as f:
        return list(json_loadlines(f, astype=Annotation))


def load_trajectories(
    path: Union[str, Path]
) -> list[tuple[str, TrajectoryData]]:
    trajs: list[tuple[str, TrajectoryData]] = []
    for root, _, files in os.walk(path, topdown=False):
        if "traj_data.json" in files:
            # File paths
            json_path = os.path.join(root, "traj_data.json")

            # Load trajectory file
            with open(json_path, "r") as f:
                traj_data = json.load(f)
            trajs.append((root, traj_data))

    print("%d num games" % len(trajs))
    return trajs
