from __future__ import annotations

from typing import Dict, List, Literal, TypedDict


Plan = Dict[str, str]


class EnvironmentState(TypedDict):
    nodes: List[Node]


Annotation = Dict[str, Dict[str, List[str]]]


class Node(TypedDict):
    id: int
    class_name: str
    category: str
    properties: List[str]
    states: List[State]
    prefab_name: str
    bounding_box: BoundingBox


class Edge(TypedDict):
    from_id: int
    relation_type: str
    to_id: int


class BoundingBox(TypedDict):
    center: List[float]
    size: List[float]


State = Literal["CLOSED", "OPEN", "ON", "OFF", "PLUGGED_IN", "PLUGGED_OUT"]
Relation = Literal["ON", "INSIDE", "CLOSE", "HOLD", "IS"]
