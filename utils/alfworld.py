from __future__ import annotations

from typing import Any, Callable, cast, overload, override

import alfworld.gen.constants as constants
import numpy as np
from alfworld.agents.controller.oracle_astar import OracleAStarAgent
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
    edge_from_tuple,
)


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

    def id2nid(self, obj_id: str) -> str | None:
        if self.__agent is None:
            raise ValueError(
                "No agent is loaded. Please check if the trajectory data has been given."
            )
        if obj_id not in self.__id2nid:
            for recep in self.__agent.receptacles.values():
                self.__nid2id[recep["num_id"]] = recep["object_id"]
                self.__id2nid[recep["object_id"]] = recep["num_id"]
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
        self, *, distance_threshold: float = 0.5
    ) -> list[Relation]:
        if self.last_event is None:
            return []

        relations: list[Relation] = []

        visible_objects = list(filter(lambda obj: obj["visible"], self.objects))

        for i, obj in enumerate(visible_objects):
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
            for target in visible_objects[(i + 1) :]:
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
            distance_threshold=distance_threshold
        )
        nodes: list[Node] = []
        edges: list[Edge] = []

        if only_visible:
            target_objects = filter(lambda elem: elem["visible"], self.objects)
        else:
            target_objects = self.objects
        for obj in target_objects:
            obj_nid = self.id2nid(obj["objectId"])
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
            from_id = self.id2nid(rel.left)
            to_id = self.id2nid(rel.right)
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

    def toggle_object(self, obj: AlfredObject | str, /, toggle: bool) -> None:
        if isinstance(obj, str):
            obj = self.get_obj_from_id(obj)
        self.step(
            {
                "action": "ToggleObjectOn",
                "objectId": obj["objectId"],
                "toggleOn": toggle,
            }
        )

    @overload
    def move_object(
        self,
        obj: AlfredObject | str,
        *,
        on: AlfredObject,
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
        on: AlfredObject | None = None,
        relative_position: tuple[float, float, float] | None = None,
    ) -> None:
        if isinstance(obj, str):
            obj = self.get_obj_from_id(obj)

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

    def remove_object(self, obj: AlfredObject) -> None:
        self.step(
            {
                "action": "RemoveFromScene",
                "objectId": obj["objectId"],
            }
        )

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
    inside_of: dict[str, str] = {}
    is_inside: dict[str, list[str]] = {}

    grabbed_ids: list[str] = []
    for edge in state["edges"]:
        if edge["relation_type"] == "INSIDE":

            if edge["to_id"] not in is_inside.keys():
                is_inside[edge["to_id"]] = []

            is_inside[edge["to_id"]].append(edge["from_id"])
            inside_of[edge["from_id"]] = edge["to_id"]

        elif "HOLDS" in edge["relation_type"]:
            if edge["from_id"] == 0:
                grabbed_ids.append(edge["to_id"])

    character_inside_ids = inside_of["agent"]
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
