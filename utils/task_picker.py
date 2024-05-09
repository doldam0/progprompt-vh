import random
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

    def __init__(
        self,
        alfworld_data_root: str | Path | None = None,
        remember: bool = False,
    ):
        if alfworld_data_root is None:
            alfworld_data_root = ALFWORLD_DATA
        self.path = Path(alfworld_data_root)
        self.__c: dict[
            str, dict[str, dict[str, dict[str, dict[int, list[Path]]]]]
        ] = {}
        self.__c_f: (
            dict[str, dict[str, dict[str, dict[str, list[Path]]]]] | None
        ) = None

        self._initialize_cache()

        self.__remember = remember
        self.__selected_tasks = set()

    @property
    def _cache(self):
        return self.__c_f if self.__c_f is not None else self.__c

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
        self.__c.clear()

        for path in self.path.iterdir():
            if path.is_dir():
                task_type, pickable, movable, receptacle, scene_num = (
                    path.name.split("-")
                )
                scene_num = int(scene_num)
                if task_type not in self.__c:
                    self.__c[task_type] = {}
                if pickable not in self.__c[task_type]:
                    self.__c[task_type][pickable] = {}
                if movable not in self.__c[task_type][pickable]:
                    self.__c[task_type][pickable][movable] = {}
                if receptacle not in self.__c[task_type][pickable][movable]:
                    self.__c[task_type][pickable][movable][receptacle] = {}
                if (
                    scene_num
                    not in self.__c[task_type][pickable][movable][receptacle]
                ):
                    self.__c[task_type][pickable][movable][receptacle][
                        scene_num
                    ] = []

                for trial_path in path.iterdir():
                    if trial_path.is_dir():
                        self.__c[task_type][pickable][movable][receptacle][
                            scene_num
                        ].append(trial_path)

    def _list_tasks(self) -> Iterable[str]:
        return self._cache.keys()

    def _list_pickables(self, task_name: str) -> Iterable[str]:
        return self._cache[task_name].keys()

    def _list_movables(self, task_name: str, pickable: str) -> Iterable[str]:
        return self._cache[task_name][pickable].keys()

    def _list_receptacles(
        self, task_name: str, pickable: str, movable: str
    ) -> Iterable[str]:
        return self._cache[task_name][pickable][movable].keys()

    def _list_scene_numbers(
        self, task_name: str, pickable: str, movable: str, receptacle: str
    ) -> Iterable[int]:
        return self.__c[task_name][pickable][movable][receptacle].keys()

    def __set_scene_num_filter(self, scene_num: int):
        self.__c_f = {}
        for task_name, pickables in self.__c.items():
            self.__c_f[task_name] = {}
            for pickable, movables in pickables.items():
                self.__c_f[task_name][pickable] = {}
                for movable, receptacles in movables.items():
                    self.__c_f[task_name][pickable][movable] = {}
                    for receptacle, traj_data in receptacles.items():
                        if scene_num in traj_data:
                            self.__c_f[task_name][pickable][movable][
                                receptacle
                            ] = traj_data[scene_num]
                    if len(self.__c_f[task_name][pickable][movable]) == 0:
                        del self.__c_f[task_name][pickable][movable]
                if len(self.__c_f[task_name][pickable]) == 0:
                    del self.__c_f[task_name][pickable]
            if len(self.__c_f[task_name]) == 0:
                del self.__c_f[task_name]

    def _list_trials(
        self,
        task_name: str,
        pickable: str,
        movable: str,
        receptacle: str,
        scene_num: int,
    ) -> list[Path]:
        return self.__c[task_name][pickable][movable][receptacle][scene_num]

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

    def pick_random(self, num_tasks: int) -> list[tuple[Path, TrajectoryData]]:
        ret: list[tuple[Path, TrajectoryData]] = []
        if not self.__remember:
            self.__selected_tasks = set()

        scene_num: int | None = None
        for _ in range(num_tasks):
            tasks = self._list_tasks()
            remained_tasks = list(set(tasks) - self.__selected_tasks)
            if len(remained_tasks) == 0:
                break
            task_type = random.choice(remained_tasks)
            self.__selected_tasks.add(task_type)

            pickables = self._list_pickables(task_type)
            pickable = random.choice(list(pickables))

            movables = self._list_movables(task_type, pickable)
            movable = random.choice(list(movables))

            receptacles = self._list_receptacles(task_type, pickable, movable)
            receptacle = random.choice(list(receptacles))

            if scene_num is None:
                scene_nums = self._list_scene_numbers(
                    task_type, pickable, movable, receptacle
                )
                scene_num_tmp = random.choice(list(scene_nums))
                self.__set_scene_num_filter(scene_num_tmp)
                scene_num = scene_num_tmp

            task_path = (
                self.path
                / f"{task_type}-{pickable}-{movable}-{receptacle}-{scene_num}"
            )
            trial = self._list_trials(
                task_type, pickable, movable, receptacle, scene_num
            )[0]

            traj_path = task_path / trial.name / "traj_data.json"
            ret.append((traj_path, load_trajectory(traj_path)))

        return ret
