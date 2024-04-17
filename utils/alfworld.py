from __future__ import annotations

import json
import os
from typing import Any, Callable, override

import alfworld.gen.constants as constants
import numpy as np
import regex
from alfworld.agents.controller.oracle_astar import OracleAStarAgent
from alfworld.env.thor_env import ThorEnv
from alfworld.gen.constants import (
    AGENT_STEP_SIZE,
    CAMERA_HEIGHT_OFFSET,
    RECORD_SMOOTHING_FACTOR,
    RENDER_CLASS_IMAGE,
    RENDER_DEPTH_IMAGE,
    RENDER_IMAGE,
    RENDER_OBJECT_IMAGE,
    VISIBILITY_DISTANCE,
)

from utils.relations import Relation, relate
from utils.types import (
    AlfredBoundingBox,
    AlfredObject,
    BoundingBox,
    Edge,
    Graph,
    Node,
    TrajectoryData,
)

_PATTERN_ACTION = r"^\[(\w+)\]"
_PATTERN_PARAMS = r"\<(.+?)\>\s*\((.+?)\)"


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


def get_object_position(obj: AlfredObject) -> tuple[float, float, float]:
    return obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]


def get_object_rotation(obj: AlfredObject) -> tuple[float, float, float]:
    return obj["rotation"]["x"], obj["rotation"]["y"], obj["rotation"]["z"]


def get_object_x_distance(obj1: AlfredObject, obj2: AlfredObject) -> float:
    return abs(obj1["position"]["x"] - obj2["position"]["x"])


def get_object_distance(obj1: AlfredObject, obj2: AlfredObject) -> float:
    x1, y1, z1 = get_object_position(obj1)
    x2, y2, z2 = get_object_position(obj2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


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

    @override
    def reset(
        self,
        trajectory_root: str,
        trajectory_data: TrajectoryData,
        *,
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

    def nid2id(self, nid: str) -> str:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        if nid not in self.__nid2id:
            for recep in self.__agent.receptacles.values():
                self.__nid2id[nid] = recep["objectId"]
                self.__id2nid[recep["objectId"]] = nid
        return self.__nid2id[nid]

    def id2nid(self, obj_id: str) -> str:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        if obj_id not in self.__id2nid:
            for recep in self.__agent.receptacles.values():
                self.__nid2id[recep["num_id"]] = recep["objectId"]
                self.__id2nid[recep["objectId"]] = recep["num_id"]
        return self.__id2nid[obj_id]

    def get_obj_from_id(self, id: str) -> AlfredObject:
        if id not in self.__id2obj:
            for obj in self.objects:
                self.__id2obj[obj["objectId"]] = obj
        return self.__id2obj[id]

    def get_recep_nid(self, obj_nid: str) -> str:
        obj_id = self.nid2id(obj_nid)
        obj = self.get_obj_from_id(obj_id)
        recep_ids = obj["receptacleObjectIds"]
        if recep_ids is None or len(recep_ids) == 0:
            return ""
        recep = self.get_obj_from_id(recep_ids[0])
        return self.id2nid(recep["objectId"])

    def extract_relations(
        self, *, distance_threshold: float = 0.5
    ) -> list[Relation]:
        if self.last_event is None:
            return []

        relations: list[Relation] = []

        visible_objects = list(filter(lambda obj: obj["visible"], self.objects))

        for i, obj in enumerate(visible_objects):
            if obj["receptacle"] and obj["receptacleObjectIds"] is not None:
                for receptacle in obj["receptacleObjectIds"]:
                    rel = relate(receptacle, "ON", obj["objectId"])
                    relations.append(rel)
            if obj["pickupable"] and obj["isPickedUp"]:
                rel = relate("agent", "HOLD", obj["objectId"])
                relations.append(rel)
            if obj["toggleable"]:
                rel = relate(
                    obj["objectId"], "IS", "ON" if obj["isToggled"] else "OFF"
                )
                relations.append(rel)
            if obj["openable"]:
                rel = relate(
                    obj["objectId"], "IS", "OPEN" if obj["isOpen"] else "CLOSED"
                )
                relations.append(rel)
            for target in visible_objects[(i + 1) :]:
                if (
                    abs(obj["position"]["x"] - target["position"]["x"])
                    > distance_threshold
                ):
                    continue
                distance = get_object_distance(obj, target)
                if distance < distance_threshold:
                    rel = relate(obj["objectId"], "CLOSE", target["objectId"])
                    relations.append(rel)

        return relations

    def environment_graph(self, *, distance_threshold: float = 0.5) -> Graph:
        relations = self.extract_relations(
            distance_threshold=distance_threshold
        )
        object_idx: dict[str, int] = {"agent": 0}
        nodes: list[Node] = []
        edges: list[Edge] = []

        visible_objects = filter(
            lambda elem: elem[1]["visible"], enumerate(self.objects)
        )
        for obj_id, obj in visible_objects:
            object_idx[obj["objectId"]] = obj_id
            node: Node = {
                "id": obj_id,
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
            if rel.left not in object_idx or rel.right not in object_idx:
                continue
            edges.append(
                {
                    "from_id": object_idx[rel.left],
                    "relation_type": rel.relation,
                    "to_id": object_idx[rel.right],
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def camera_count(self) -> int:
        return len(self.metadata["cameraHorizons"])

    def camera_image(
        self,
        idx: int | list[int],
        image_width: int = 300,
        image_height: int = 300,
    ) -> list[np.ndarray]:
        if isinstance(idx, int):
            idx = [idx]
        return [
            self.event().cv2img(
                self.metadata["cameraHorizons"][i]["cameraHorizon"],
                width=image_width,
                height=image_height,
            )
            for i in idx
        ]

    def __render_single_script(self, script: str) -> None:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        return self.__agent.step(script)

    def render_script(self, script: list[str] | str) -> None:
        if isinstance(script, str):
            script = [script]
        for s in script:
            self.__render_single_script(s)


def get_visible_nodes(graph: Graph) -> Graph:
    # Obtains partial observation from the perspective of agent_id
    # That is, objects inside the same room as agent_id and not inside closed containers
    # NOTE: Assumption is that the graph has an inside transition that is not transitive
    state = graph
    id2node = {node["id"]: node for node in state["nodes"]}
    rooms_ids = [
        node["id"] for node in graph["nodes"] if node["category"] == "Rooms"
    ]

    # find character
    inside_of: dict[int, int] = {}
    is_inside: dict[int, list[int]] = {}

    grabbed_ids: list[int] = []
    for edge in state["edges"]:
        if edge["relation_type"] == "INSIDE":

            if edge["to_id"] not in is_inside.keys():
                is_inside[edge["to_id"]] = []

            is_inside[edge["to_id"]].append(edge["from_id"])
            inside_of[edge["from_id"]] = edge["to_id"]

        elif "HOLDS" in edge["relation_type"]:
            if edge["from_id"] == 0:
                grabbed_ids.append(edge["to_id"])

    character_inside_ids = inside_of[0]
    room_id = character_inside_ids

    object_in_room_ids = is_inside[room_id]

    # Some object are not directly in room, but we want to add them
    curr_objects = list(object_in_room_ids)
    while len(curr_objects) > 0:
        objects_inside = []
        for curr_obj_id in curr_objects:
            new_inside = (
                is_inside[curr_obj_id]
                if curr_obj_id in is_inside.keys()
                else []
            )
            objects_inside += new_inside

        object_in_room_ids += list(objects_inside)
        curr_objects = list(objects_inside)

    # Only objects that are inside the room and not inside something closed
    # TODO: this can be probably speed up if we can ensure that all objects are either closed or open
    object_hidden = (
        lambda ido: inside_of[ido] not in rooms_ids
        and "OPEN" not in id2node[inside_of[ido]]["states"]
    )
    observable_object_ids = [
        object_id
        for object_id in object_in_room_ids
        if not object_hidden(object_id)
    ] + rooms_ids
    observable_object_ids += grabbed_ids

    partilly_observable_state: Graph = {
        "edges": [
            edge
            for edge in state["edges"]
            if edge["from_id"] in observable_object_ids
            and edge["to_id"] in observable_object_ids
        ],
        "nodes": [id2node[id_node] for id_node in observable_object_ids],
    }

    return partilly_observable_state
