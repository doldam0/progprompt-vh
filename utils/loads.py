import json
from pathlib import Path
from typing import Any, Dict, Iterator, TextIO, Type, TypeVar, Union

from utils.types import Annotation, EnvironmentState, Plan

T = TypeVar("T")


def json_loadlines(f: TextIO, astype: Type[T] = Dict[str, Any]) -> Iterator[T]:
    while line := f.readline():
        yield json.loads(line)


def load_plan(filename: Union[str, Path]) -> Plan:
    with open(filename, "r") as f:
        return json.load(f)


def load_environment_states(
    filename: Union[str, Path]
) -> Iterator[EnvironmentState]:
    with open(filename, "r") as f:
        return json_loadlines(f, astype=EnvironmentState)


def load_annotations(filename: Union[str, Path]) -> Iterator[Annotation]:
    with open(filename, "r") as f:
        return json_loadlines(f, astype=Annotation)
