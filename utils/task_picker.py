from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from alfworld.info import ALFWORLD_DATA

from utils import prompt
from utils.loads import load_trajectory
from utils.types import TrajectoryData


@dataclass
class TaskInformation:
    path: str
    scene_name: str


class AlfWorldTaskPicker:
    _INTERFACE_STYLE = prompt.style_from_dict(
        {
            "separator": "#6C6C6C",
            "questionmark": "#FF9D00 bold",
            "selected": "#5F819D",
            "pointer": "#FF9D00 bold",
            "instruction": "",  # default
            "answer": "#5F819D bold",
            "question": "",
        }
    )

    def __init__(self, alfworld_data_root: str | Path | None = None):
        if alfworld_data_root is None:
            alfworld_data_root = ALFWORLD_DATA
        self.path = Path(alfworld_data_root)
        self.__task_path_cache: dict[
            str, dict[str, dict[str, dict[str, dict[int, list[Path]]]]]
        ] = {}
        self.__task_path_cache_filtered: (
            dict[str, dict[str, dict[str, dict[str, list[Path]]]]] | None
        ) = None

        self._initialize_cache()

    @property
    def __cache(self):
        return (
            self.__task_path_cache_filtered
            if self.__task_path_cache_filtered is not None
            else self.__task_path_cache
        )

    def pick(
        self,
        task_type: str,
        *args: str | None,
        task_num: int | None = None,
        trial_num: int = 0,
    ) -> tuple[Path, TrajectoryData]:
        if len(args) < 3:
            args += (None,) * (3 - len(args))

        top_dirname = f"{task_type.lower().replace(' ', '_')}-{args[0]}-{args[1]}-{args[2]}"
        if task_num is not None:
            top_dirname += f"-{task_num}"
        else:
            for path in self.path.iterdir():
                if path.is_dir() and path.name.startswith(top_dirname):
                    top_dirname = path.name
                    break

        traj_path = self.path / top_dirname
        if not traj_path.exists():
            raise FileNotFoundError(f"Task not found: {traj_path}")
        if not traj_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {traj_path}")

        for i, path in enumerate(traj_path.iterdir()):
            if i == trial_num:
                traj_path = path
                break

        traj_path = traj_path / "traj_data.json"
        return traj_path, load_trajectory(traj_path)

    def _initialize_cache(self) -> None:
        self.__task_path_cache.clear()

        for path in self.path.iterdir():
            if path.is_dir():
                task_type, pickable, movable, receptacle, scene_num = (
                    path.name.split("-")
                )
                scene_num = int(scene_num)
                if task_type not in self.__task_path_cache:
                    self.__task_path_cache[task_type] = {}
                if pickable not in self.__task_path_cache[task_type]:
                    self.__task_path_cache[task_type][pickable] = {}
                if movable not in self.__task_path_cache[task_type][pickable]:
                    self.__task_path_cache[task_type][pickable][movable] = {}
                if (
                    receptacle
                    not in self.__task_path_cache[task_type][pickable][movable]
                ):
                    self.__task_path_cache[task_type][pickable][movable][
                        receptacle
                    ] = {}
                if (
                    scene_num
                    not in self.__task_path_cache[task_type][pickable][movable][
                        receptacle
                    ]
                ):
                    self.__task_path_cache[task_type][pickable][movable][
                        receptacle
                    ][scene_num] = []

                for trial_path in path.iterdir():
                    if trial_path.is_dir():
                        self.__task_path_cache[task_type][pickable][movable][
                            receptacle
                        ][scene_num].append(trial_path)

    def _list_tasks(self) -> Iterable[str]:
        return self.__cache.keys()

    def _list_pickables(self, task_name: str) -> Iterable[str]:
        return self.__cache[task_name].keys()

    def _list_movables(self, task_name: str, pickable: str) -> Iterable[str]:
        return self.__cache[task_name][pickable].keys()

    def _list_receptacles(
        self, task_name: str, pickable: str, movable: str
    ) -> Iterable[str]:
        return self.__cache[task_name][pickable][movable].keys()

    def _list_scene_numbers(
        self, task_name: str, pickable: str, movable: str, receptacle: str
    ) -> Iterable[int]:
        return self.__task_path_cache[task_name][pickable][movable][
            receptacle
        ].keys()

    def __set_scene_num_filter(self, scene_num: int):
        self.__task_path_cache_filtered = {}
        for task_name, pickables in self.__task_path_cache.items():
            self.__task_path_cache_filtered[task_name] = {}
            for pickable, movables in pickables.items():
                self.__task_path_cache_filtered[task_name][pickable] = {}
                for movable, receptacles in movables.items():
                    self.__task_path_cache_filtered[task_name][pickable][
                        movable
                    ] = {}
                    for receptacle, traj_data in receptacles.items():
                        if scene_num in traj_data:
                            self.__task_path_cache_filtered[task_name][
                                pickable
                            ][movable][receptacle] = traj_data[scene_num]
                    if (
                        len(
                            self.__task_path_cache_filtered[task_name][
                                pickable
                            ][movable]
                        )
                        == 0
                    ):
                        del self.__task_path_cache_filtered[task_name][
                            pickable
                        ][movable]
                if (
                    len(self.__task_path_cache_filtered[task_name][pickable])
                    == 0
                ):
                    del self.__task_path_cache_filtered[task_name][pickable]
            if len(self.__task_path_cache_filtered[task_name]) == 0:
                del self.__task_path_cache_filtered[task_name]

    def _list_trials(
        self,
        task_name: str,
        pickable: str,
        movable: str,
        receptacle: str,
        scene_num: int,
    ) -> list[Path]:
        return self.__task_path_cache[task_name][pickable][movable][receptacle][
            scene_num
        ]

    def pick_interactive(self) -> list[tuple[Path, TrajectoryData]]:
        ret: list[tuple[Path, TrajectoryData]] = []

        i = 1
        scene_num: int | None = None
        while True:
            tasks = self._list_tasks()
            task_type = prompt.list(
                "Which task type do you want to pick?",
                list(sorted(tasks)) + (["exit"] if i > 1 else []),
                style=self._INTERFACE_STYLE,
                clear=True,
                qmark=f"[{i}]",
            )

            if task_type == "exit":
                break

            pickables = self._list_pickables(task_type)
            pickable = prompt.list(
                "Which pickable do you want to pick?",
                sorted(pickables),
                style=self._INTERFACE_STYLE,
                clear=True,
                qmark=f"[{i}]",
            )

            movables = self._list_movables(task_type, pickable)
            movable = prompt.list(
                "Which movable do you want to pick?",
                sorted(movables),
                style=self._INTERFACE_STYLE,
                clear=True,
                qmark=f"[{i}]",
            )

            receptacles = self._list_receptacles(task_type, pickable, movable)
            receptacle = prompt.list(
                "Which receptacle do you want to pick?",
                sorted(receptacles),
                style=self._INTERFACE_STYLE,
                clear=True,
                qmark=f"[{i}]",
            )

            if scene_num is None:
                scene_nums = self._list_scene_numbers(
                    task_type, pickable, movable, receptacle
                )
                scene_num = prompt.list(
                    "Which room number do you want to pick?",
                    sorted(scene_nums),
                    style=self._INTERFACE_STYLE,
                    clear=True,
                    qmark=f"[{i}]",
                )

                self.__set_scene_num_filter(scene_num)

            task_path = (
                self.path
                / f"{task_type}-{pickable}-{movable}-{receptacle}-{scene_num}"
            )
            trial = self._list_trials(
                task_type, pickable, movable, receptacle, scene_num
            )[0]

            traj_path = task_path / trial.name / "traj_data.json"
            ret.append((traj_path, load_trajectory(traj_path)))

            i += 1

        return ret