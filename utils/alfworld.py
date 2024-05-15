from __future__ import annotations

import os.path as osp
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, cast, overload, override

import alfworld.agents
import alfworld.gen.constants as constants
import numpy as np
from alfworld.agents.controller.oracle_astar import OracleAStarAgent
from alfworld.env.tasks import get_task
from alfworld.env.thor_env import ThorEnv

from utils.relations import Relation, Relations, relate
from utils.types import (
    AlfredBoundingBox,
    AlfredObject,
    BoundingBox,
    Edge,
    Graph,
    Node,
    TrajectoryData,
)

ACTIONS = {
    "goto": "go to {0}",
    "take": "take {0} from {1}",
    "put": "put {0} on {1}",
    "open": "open {0}",
    "close": "close {0}",
    "toggle": "toggle {0}",
    "heat": "heat {0} with {1}",
    "cool": "cool {0} with {1}",
    "clean": "clean {0} with {1}",
    "inventory": "inventory",
}


def bbox_from_alfred(alfred_bbox: AlfredBoundingBox) -> BoundingBox:
    top_left = alfred_bbox["objectBoundsCorners"][0]
    bottom_right = alfred_bbox["objectBoundsCorners"][6]

    return BoundingBox(
        center=[
            (top_left["x"] + bottom_right["x"]) / 2,
            (top_left["y"] + bottom_right["y"]) / 2,
            (top_left["z"] + bottom_right["z"]) / 2,
        ],
        size=[
            bottom_right["x"] - top_left["x"],
            bottom_right["y"] - top_left["y"],
            bottom_right["z"] - top_left["z"],
        ],
    )


def add_vector3d(
    v1: tuple[float, float, float], v2: tuple[float, float, float]
) -> tuple[float, float, float]:
    return v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]


def get_object_position(obj: AlfredObject) -> tuple[float, float, float]:
    return obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]


def get_object_rotation(obj: AlfredObject) -> tuple[float, float, float]:
    return obj["rotation"]["x"], obj["rotation"]["y"], obj["rotation"]["z"]


def get_object_x_distance(obj1: AlfredObject, obj2: AlfredObject) -> float:
    return abs(obj1["position"]["x"] - obj2["position"]["x"])


def get_distance(
    p1: tuple[float, float, float], p2: tuple[float, float, float]
) -> float:
    return (
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    ) ** 0.5


def get_object_distance(obj1: AlfredObject, obj2: AlfredObject) -> float:
    x1, y1, z1 = get_object_position(obj1)
    x2, y2, z2 = get_object_position(obj2)
    return get_distance((x1, y1, z1), (x2, y2, z2))


class CustomThorEnv(ThorEnv):
    @override
    def __init__(
        self,
        /,
        x_display: str = constants.X_DISPLAY,
        player_screen_height: int = constants.DETECTION_SCREEN_HEIGHT,
        player_screen_width: int = constants.DETECTION_SCREEN_WIDTH,
        quality: str = "MediumCloseFitShadows",
        build_path: str | None = constants.BUILD_PATH,
        save_frames_to_disk: bool = False,
        save_frames_path: str = "./",
        smooth_nav: bool = False,
        callback: Callable[[CustomThorEnv], Any] | None = None,
        load_receps: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            x_display,
            player_screen_height,
            player_screen_width,
            quality,
            build_path,
            save_frames_to_disk,
            save_frames_path,
            smooth_nav,
        )
        self.__callback = callback
        self.__load_receps = load_receps
        self.__debug = debug
        self.__nid2id: dict[str, str] = {}
        self.__id2nid: dict[str, str] = {}
        self.__id2obj: dict[str, AlfredObject] = {}
        self.__initial_states: dict[str, tuple[float, float, float]] = {}

    @override
    def reset(
        self,
        trajectory_root: str,
        trajectory_data: TrajectoryData,
        *,
        reward_config_path: str | Path | None = None,
        grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
        camera_y=constants.CAMERA_HEIGHT_OFFSET,
        render_image=constants.RENDER_IMAGE,
        render_depth_image=constants.RENDER_DEPTH_IMAGE,
        render_class_image=constants.RENDER_CLASS_IMAGE,
        render_object_image=constants.RENDER_OBJECT_IMAGE,
        visibility_distance=constants.VISIBILITY_DISTANCE,
        reward_type="dense",
    ):
        self.__nid2id.clear()
        self.__id2nid.clear()
        self.__id2obj.clear()

        scene_num = trajectory_data["scene"]["scene_num"]
        object_poses = trajectory_data["scene"]["object_poses"]
        dirty_and_empty = trajectory_data["scene"]["dirty_and_empty"]
        object_toggles = trajectory_data["scene"]["object_toggles"]
        scene_name = "FloorPlan%d" % scene_num

        super().reset(
            scene_name,
            grid_size,
            camera_y,
            render_image,
            render_depth_image,
            render_class_image,
            render_object_image,
            visibility_distance,
        )
        self.restore_scene(object_poses, object_toggles, dirty_and_empty)
        self.step(dict(trajectory_data["scene"]["init_action"]))

        self.__agent = OracleAStarAgent(
            env=self,
            traj_data=trajectory_data,
            traj_root=trajectory_root,
            load_receps=self.__load_receps,
            debug=self.__debug,
        )

        if reward_config_path is None:
            reward_config_path = osp.join(
                next(iter(alfworld.agents.__path__)),
                "config",
                "rewards.json",
            )
        self._args = Namespace()
        self._args.reward_config = reward_config_path
        self.set_task(trajectory_data, self._args, reward_type=reward_type)

        self.__initial_states = {
            obj["objectId"]: get_object_position(obj)
            for obj in self.metadata["objects"]
        }

    @override
    def step(self, action, smooth_nav=False):
        event = super().step(action, smooth_nav)
        if self.__callback is not None:
            self.__callback(self)
        return event

    def event(self):
        if self.last_event is None:
            raise ValueError("No event has been recorded yet")
        return self.last_event

    @property
    def metadata(self):
        return self.event().metadata

    @property
    def objects(self) -> list[AlfredObject]:
        return self.metadata["objects"]

    @property
    def agent(self) -> OracleAStarAgent:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        return self.__agent

    def get_agent_position(self) -> tuple[float, float, float]:
        pos = self.metadata["agent"]["position"]
        return pos["x"], pos["y"], pos["z"]

    def nid2id(self, nid: str) -> str:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        if nid not in self.__nid2id:
            for obj in {
                **self.__agent.objects,
                **self.__agent.receptacles,
            }.values():
                self.__nid2id[obj["num_id"]] = obj["object_id"]
                self.__id2nid[obj["object_id"]] = obj["num_id"]
        return self.__nid2id[nid]

    def id2nid(self, obj_id: str) -> str | None:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        if obj_id not in self.__id2nid:
            for obj in {
                **self.__agent.objects,
                **self.__agent.receptacles,
            }.values():
                self.__nid2id[obj["num_id"]] = obj["object_id"]
                self.__id2nid[obj["object_id"]] = obj["num_id"]
        return self.__id2nid[obj_id] if obj_id in self.__id2nid else None

    def get_obj_from_id(self, id: str) -> AlfredObject:
        if id not in self.__id2obj:
            for obj in self.objects:
                self.__id2obj[obj["objectId"]] = obj
        return self.__id2obj[id]

    def get_recep_nid(self, obj_nid: str) -> str | None:
        obj_id = self.nid2id(obj_nid)
        obj = self.get_obj_from_id(obj_id)
        recep_ids = obj["receptacleObjectIds"]
        if recep_ids is None or len(recep_ids) == 0:
            return ""
        recep = self.get_obj_from_id(recep_ids[0])
        return self.id2nid(recep["objectId"])

    def extract_relations(
        self, *, only_visible: bool = False, distance_threshold: float = 0.5
    ) -> list[Relation]:
        if self.last_event is None:
            return []

        relations: list[Relation] = []

        if only_visible:
            target_objects = list(
                filter(lambda obj: obj["visible"], self.objects)
            )
        else:
            target_objects = self.objects

        for i, obj in enumerate(target_objects):
            if obj["receptacle"] and obj["receptacleObjectIds"] is not None:
                for receptacle in obj["receptacleObjectIds"]:
                    rel = relate(receptacle, "on", obj["objectId"])
                    relations.append(rel)

            if obj["pickupable"] and obj["isPickedUp"]:
                rel = relate("agent", "hold", obj["objectId"])
                relations.append(rel)

            if obj["toggleable"]:
                rel = relate(
                    obj["objectId"], "is", "on" if obj["isToggled"] else "off"
                )
                relations.append(rel)

            if obj["openable"]:
                rel = relate(
                    obj["objectId"], "is", "open" if obj["isOpen"] else "closed"
                )
                relations.append(rel)

            obj_pos = get_object_position(obj)
            agent_pos = self.get_agent_position()
            if get_distance(obj_pos, agent_pos) < distance_threshold:
                rel = relate("agent", "close", obj["objectId"])
                relations.append(rel)

            for target in target_objects[(i + 1) :]:
                if (
                    abs(obj["position"]["x"] - target["position"]["x"])
                    > distance_threshold
                ):
                    continue
                distance = get_object_distance(obj, target)
                if distance < distance_threshold:
                    rel = relate(obj["objectId"], "close", target["objectId"])
                    relations.append(rel)

        return relations

    def environment_graph(
        self, *, only_visible: bool = True, distance_threshold: float = 0.5
    ) -> Graph:
        relations = self.extract_relations(
            only_visible=only_visible, distance_threshold=distance_threshold
        )
        nodes: list[Node] = []
        edges: list[Edge] = []

        if only_visible:
            target_objects = filter(lambda elem: elem["visible"], self.objects)
        else:
            target_objects = self.objects
        for obj in target_objects:
            obj_nid = obj["objectId"]
            if obj_nid is None:
                continue
            node: Node = {
                "id": obj_nid,
                "category": obj["objectType"],
                "class_name": obj["objectType"],
                "prefab_name": obj["name"],
                "states": [],
                "properties": [],
            }
            if (bbox := obj["objectBounds"]) is not None:
                node["bounding_box"] = bbox_from_alfred(bbox)
            nodes.append(node)

        for rel in relations:
            from_id = rel.left
            to_id = rel.right
            if from_id is None or to_id is None:
                continue
            edges.append(
                {
                    "from_id": from_id,
                    "relation_type": rel.relation,
                    "to_id": to_id,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def camera_count(self) -> int:
        return 1

    def camera_image(self) -> np.ndarray:
        return self.event().cv2img

    def __render_single_script(self, script: str) -> bool:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        self.__agent.step(script)
        return self.__agent.feedback != "Nothing happens."

    def render_script(self, script: list[str] | str) -> int:
        if isinstance(script, str):
            script = [script]
        return sum(int(self.__render_single_script(s)) for s in script)

    def toggle_object(
        self, obj: AlfredObject | str, /, toggle: bool | None = None
    ) -> None:
        if isinstance(obj, str):
            if len(obj.split()) > 1:
                obj = self.nid2id(obj)
            obj = self.get_obj_from_id(obj)

        if toggle is None:
            toggle = not obj["isToggled"]

        if obj["toggleable"]:
            self.step(
                {
                    "action": "ToggleObject" + ("On" if toggle else "Off"),
                    "objectId": obj["objectId"],
                }
            )
        else:
            self.init_object_position(obj)

    @overload
    def move_object(
        self,
        obj: AlfredObject | str,
        *,
        on: AlfredObject | str,
        relative_position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float] | None = None,
    ) -> None: ...

    @overload
    def move_object(
        self,
        obj: AlfredObject | str,
        *,
        position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float] | None = None,
    ) -> None: ...

    def move_object(
        self,
        obj: AlfredObject | str,
        *,
        position: tuple[float, float, float] | None = None,
        rotation: tuple[float, float, float] | None = None,
        on: AlfredObject | str | None = None,
        relative_position: tuple[float, float, float] | None = None,
    ) -> None:
        if isinstance(obj, str):
            if len(obj.split()) > 1:
                obj = self.nid2id(obj)
            obj = self.get_obj_from_id(obj)

        if isinstance(on, str):
            if len(on.split()) > 1:
                on = self.nid2id(on)
            on = self.get_obj_from_id(on)

        if position is None:
            if on is not None:
                if relative_position is not None:
                    relative_position = relative_position
                else:
                    relative_position = (0, 0, 0)
                position = add_vector3d(
                    get_object_position(on), relative_position
                )
            else:
                position = get_object_position(obj)

        if rotation is None:
            rotation = get_object_rotation(obj)

        self.step(
            {
                "action": "SetObjectPoses",
                "objectPoses": [
                    {
                        "objectName": obj["name"],
                        "position": {
                            "x": position[0],
                            "y": position[1],
                            "z": position[2],
                        },
                        "rotation": {
                            "x": rotation[0],
                            "y": rotation[1],
                            "z": rotation[2],
                        },
                    }
                ]
                + [
                    {
                        "objectName": elem["name"],
                        "position": elem["position"],
                        "rotation": elem["rotation"],
                    }
                    for elem in self.metadata["objects"]
                    if elem["pickupable"] and elem["name"] != obj["name"]
                ],
            }
        )

    def remove_object(self, obj: AlfredObject | str) -> None:
        if isinstance(obj, str):
            if len(obj.split()) > 1:
                obj = self.nid2id(obj)
            obj = self.get_obj_from_id(obj)

        self.step(
            {
                "action": "RemoveFromScene",
                "objectId": obj["objectId"],
            }
        )

    def init_object_position(self, obj: AlfredObject | str) -> None:
        if isinstance(obj, str):
            if len(obj.split()) > 1:
                obj = self.nid2id(obj)
            obj = self.get_obj_from_id(obj)

        self.move_object(obj, position=self.__initial_states[obj["objectId"]])

    def check_conditions(
        self,
        conditions: list[tuple[str, str, str]] | list[Relation],
        *,
        distance_threshold: float = 0.5,
    ) -> bool:
        if all(isinstance(cond, tuple) for cond in conditions):
            conditions = [
                Relation(cond[0], Relations(cond[1]), cond[2])  # type: ignore
                for cond in conditions
            ]

        conditions = cast(list[Relation], conditions)
        for cond in conditions:
            left = self.get_obj_from_id(cond.left)
            right = self.get_obj_from_id(cond.right)

            match cond.relation:
                case "is":
                    match cond.right:
                        case "open":
                            if not left["isOpen"]:
                                return False
                        case "closed":
                            if left["isOpen"]:
                                return False
                        case "on":
                            if not left["isToggled"]:
                                return False
                        case "off":
                            if left["isToggled"]:
                                return False
                        case _:
                            raise ValueError(f"Unknown state: {cond}")
                case "hold":
                    if not right["isPickedUp"]:
                        return False
                case "on":
                    if not left["isToggled"]:
                        return False
                case "close":
                    if get_object_distance(left, right) > distance_threshold:
                        return False
                case _:
                    raise ValueError(f"Unknown condition: {cond}")
        return True


class MultipleTaskThorEnv(CustomThorEnv):
    @override
    def reset(
        self,
        trajectories: list[tuple[Path, TrajectoryData]],
        *,
        reward_config_path: str | Path | None = None,
        grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
        camera_y=constants.CAMERA_HEIGHT_OFFSET,
        render_image=constants.RENDER_IMAGE,
        render_depth_image=constants.RENDER_DEPTH_IMAGE,
        render_class_image=constants.RENDER_CLASS_IMAGE,
        render_object_image=constants.RENDER_OBJECT_IMAGE,
        visibility_distance=constants.VISIBILITY_DISTANCE,
        reward_type="dense",
    ):
        path, traj_data = trajectories[0]
        root = str(path.parent)

        object_poses = list(
            {
                f'{e["objectName"]}'
                f'|{int(e["position"]["x"])}'
                f'|{int(e["position"]["y"])}'
                f'|{int(e["position"]["z"])}': e
                for e in sum(
                    (traj["scene"]["object_poses"] for _, traj in trajectories),
                    start=[],
                )
            }.values()
        )
        object_toggles = list(
            {
                e["objectType"]: e
                for e in sum(
                    (
                        traj["scene"]["object_toggles"]
                        for _, traj in trajectories
                    ),
                    start=[],
                )
            }.values()
        )

        traj_data["scene"]["object_poses"] = object_poses
        traj_data["scene"]["object_toggles"] = object_toggles

        super().reset(
            root,
            traj_data,
            reward_config_path=reward_config_path,
            grid_size=grid_size,
            camera_y=camera_y,
            render_image=render_image,
            render_depth_image=render_depth_image,
            render_class_image=render_class_image,
            render_object_image=render_object_image,
            visibility_distance=visibility_distance,
            reward_type=reward_type,
        )

        self._tasks = [
            (
                traj_root,
                get_task(
                    traj_data["task_type"],
                    traj_data,
                    self,
                    self._args,
                    reward_type=reward_type,
                ),
            )
            for traj_root, traj_data in trajectories
        ]

    @override
    def get_goal_satisfied(self):
        if self._tasks is None:
            raise ValueError("No tasks have been loaded yet")
        return all(
            task.goal_satisfied(self.last_event) for _, task in self._tasks
        )

    @overload
    def get_which_goal_satisfied(self, task_id: str) -> bool: ...

    @overload
    def get_which_goal_satisfied(self) -> list[tuple[Path, bool]]: ...

    def get_which_goal_satisfied(
        self, task_id: str | None = None
    ) -> bool | list[tuple[Path, bool]]:
        if self._tasks is None:
            raise ValueError("No tasks have been loaded yet")
        if task_id is not None:
            for _, task in self._tasks:
                if task_id == task.traj["task_id"]:
                    return task.goal_satisfied(self.last_event)
        return [
            (path, task.goal_satisfied(self.last_event))
            for path, task in self._tasks
        ]


def combine_graphs(graphs: Graph | list[Graph], *others: Graph) -> Graph:
    if not isinstance(graphs, list):
        graphs = [graphs]
    graphs.extend(others)

    nodes: dict[str, Node] = {}
    edges: dict[str, Edge] = {}

    for graph in graphs:
        for node in graph["nodes"]:
            nodes[node["id"]] = node
        for edge in graph["edges"]:
            edges[f"{edge['from_id']}_{edge['to_id']}"] = edge

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


def validate_graph(graph: Graph) -> Graph:
    nodes = graph["nodes"]
    edges = graph["edges"]

    return {
        "nodes": nodes,
        "edges": [
            {
                "from_id": edge["from_id"],
                "relation_type": edge["relation_type"],
                "to_id": edge["to_id"],
            }
            for edge in edges
            if (edge["from_id"].find("|") == -1 or edge["from_id"] in nodes)
            and (edge["to_id"].find("|") == -1 or edge["to_id"] in nodes)
        ],
    }
