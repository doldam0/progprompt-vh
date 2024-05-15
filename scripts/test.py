#!/usr/bin/env python

from __future__ import annotations

import os.path as osp
import sys
from pathlib import Path

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import argparse

from alfworld.gen import constants
from alfworld.info import ALFWORLD_DATA

from utils.alfworld import MultipleTaskThorEnv
from utils.loads import load_trajectory
from utils.relations import Relation

conditions: list[Relation] = []


def main(args: TestArgument):
    # start THOR
    env = MultipleTaskThorEnv()

    # load traj_data
    if args.problem is not None:
        root = Path(args.problem)
        json_file = root / "traj_data.json"
        traj_data = load_trajectory(json_file)
        tasks = [(root, traj_data)]
    else:
        from utils.task_picker import AlfWorldTaskPicker

        task_picker = AlfWorldTaskPicker()
        tasks = task_picker.pick_interactive()

    # reset environment
    env.reset(tasks, reward_config_path=args.reward_config)

    print(env.agent.feedback)
    while True:
        cmd = input()
        if cmd == "ipdb":
            from ipdb import set_trace

            set_trace()
            continue

        if cmd.startswith("\\"):
            cmd_type, *cmd_args = cmd.split()
            match cmd_type[1:]:
                case "toggle":
                    if len(cmd_args) < 2:
                        print("Usage: \\toggle OBJECT_ID")
                        continue

                    env.toggle_object(" ".join(cmd_args[:2]))
                    print(f"Toggled {cmd_args[1]}\n")
                case "move":
                    if len(cmd_args) < 6 or cmd_args[2] not in ["to", "on"]:
                        print("Usage: \\move OBJECT_ID to X Y Z")
                        print("Usage: \\move OBJECT_ID on OBJECT_ID")
                        continue

                    match cmd_args[2]:
                        case "on":
                            env.move_object(
                                " ".join(cmd_args[:2]),
                                on=" ".join(cmd_args[3:5]),
                            )
                            print(f"Moved {cmd_args[1]} on {cmd_args[4]}\n")
                        case "to":
                            env.move_object(
                                " ".join(cmd_args[:2]),
                                position=(
                                    float(cmd_args[3]),
                                    float(cmd_args[4]),
                                    float(cmd_args[5]),
                                ),
                            )
                            print(
                                f"Moved {cmd_args[1]} to {cmd_args[3]} {cmd_args[4]} {cmd_args[5]}\n"
                            )
                case "remove":
                    if len(cmd_args) < 2:
                        print("Usage: \\remove OBJECT_ID")
                        continue

                    env.remove_object(" ".join(cmd_args[:2]))
                    print(f"Removed {cmd_args[1]}\n")
        else:
            graph = env.environment_graph()
            print(graph)

            env.render_script(cmd)
            if not args.debug:
                print(env.agent.feedback)

            done = env.get_goal_satisfied()
            if done:
                print("You won!")
                break

        print("Conditions:")
        conditions = env.get_which_goal_satisfied()
        for traj_root, cond in conditions:
            print(f"{traj_root.parent.parent.name}: {cond}")

        print()


class TestArgument(argparse.Namespace):
    """Play the abstract text version of an ALFRED environment."""

    """Path to a folder containing PDDL and traj_data files."""
    problem: str | None

    """Print debug information."""
    debug: bool

    """Load receps."""
    load_receps: bool

    """Path to the reward configuration file."""
    reward_config: str | None

    """X display to use."""
    x_display: int


def parse_args() -> TestArgument:
    description = "Play the abstract text version of an ALFRED environment."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "problem",
        nargs="?",
        default=None,
        help="Path to a folder containing PDDL and traj_data files. "
        f"Default: pick a problem in {ALFWORLD_DATA} with interactive input.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug information."
    )
    parser.add_argument(
        "--load-receps", action="store_true", help="Load receps."
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        required=False,
        help="Path to the reward configuration file.",
    )
    parser.add_argument("--x-display", default=0, help="X display to use.")
    return parser.parse_args(namespace=TestArgument())


if __name__ == "__main__":
    args = parse_args()

    constants.X_DISPLAY = str(args.x_display)

    main(args)
