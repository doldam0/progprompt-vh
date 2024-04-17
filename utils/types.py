from __future__ import annotations

from typing import Any, NewType, NotRequired, TypedDict

Plan = dict[str, str]

TrajectoryData = NewType("TrajectoryData", dict[str, Any])


class Graph(TypedDict):
    nodes: list[Node]
    edges: list[Edge]


Annotation = dict[str, dict[str, list[str]]]


class Node(TypedDict):
    id: int
    class_name: str
    category: str
    properties: list[str]
    states: list[str]
    prefab_name: str
    bounding_box: NotRequired[BoundingBox]


class Edge(TypedDict):
    from_id: int
    relation_type: str
    to_id: int


class BoundingBox(TypedDict):
    center: list[float]
    size: list[float]


class AlfredTrajectoryImage(TypedDict):
    """High-level trajectory image"""

    """Low-level action index"""
    high_idx: int

    """High-level action index"""
    low_idx: int

    """Image filename"""
    image_name: str


class PDDLParams(TypedDict):
    """High-level PDDL parameters"""

    """Movable receptacle"""
    mrecep_target: str

    """Whether the object should be sliced"""
    object_sliced: bool

    """Object to be picked up"""
    object_target: str

    """Receptacle for the object"""
    parent_target: str

    """Toggle object"""
    toggle_target: str


class HighPDDL(TypedDict):
    """High-level PDDL action"""

    discrete_action: dict
    high_idx: int
    planer_action: dict


class AlfredPlan(TypedDict):
    """High-level PDDL plan"""

    high_pddl: list[HighPDDL]
    low_actions: list[dict]


class AlfredScene(TypedDict):
    """Scene information"""

    dirty_and_empty: bool
    floor_plan: str
    init_action: dict
    object_poses: list[dict]
    object_toggles: list[dict]
    random_seed: int
    scene_num: int


class AlfredTrajectory(TypedDict):
    images: list[AlfredTrajectoryImage]
    pddl_params: PDDLParams
    plan: AlfredPlan
    scene: AlfredScene
    task_id: str
    task_type: str
    turk_annotations: dict


class Vector3(TypedDict):
    """Object position"""

    x: float
    y: float
    z: float


class AlfredBoundingBox(TypedDict):
    objectBoundsCorners: list[Vector3]


class AlfredObject(TypedDict):
    """Object information"""

    name: str
    position: Vector3
    rotation: Vector3
    cameraHorizon: float
    visible: bool
    receptacle: bool
    toggleable: bool
    isToggled: bool
    breakable: bool
    isBroken: bool
    canFillWithLiquid: bool
    isFilledWithLiquid: bool
    dirtyable: bool
    isDirty: bool
    canBeUsedUp: bool
    isUsedUp: bool
    cookable: bool
    isCooked: bool
    objectTemperature: str
    canChangeTempToHot: bool
    canChangeTempToCold: bool
    sliceable: bool
    isSliced: bool
    openable: bool
    isOpen: bool
    pickupable: bool
    isPickedUp: bool
    mass: float
    salientMaterials: list[str]
    receptacleObjectIds: list[str] | None
    distance: float
    objectType: str
    objectId: str
    parentReceptacle: str | None
    parentReceptacles: list[str] | None
    currentTime: float
    isMoving: bool
    objectBounds: AlfredBoundingBox
