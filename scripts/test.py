#!/usr/bin/env python

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from os.path import join as pjoin

import alfworld.agents
from alfworld.gen import constants
from alfworld.info import ALFWORLD_DATA

from utils.alfworld import CustomThorEnv
from utils.relations import Relation

conditions: list[Relation] = []


def main(args: TestArgument):
    print(f"Playing '{args.problem}'.")

    # start THOR
    env = CustomThorEnv()

    # load traj_data
    root = args.problem
    json_file = os.path.join(root, "traj_data.json")
    with open(json_file, "r") as f:
        traj_data = json.load(f)

    # reset environment
    env.reset(root, traj_data)

    print(env.agent.feedback)
    while True:
        cmd = input()
        if cmd == "ipdb":
            from ipdb import set_trace

            set_trace()
            continue

        env.render_script(cmd)
        if not args.debug:
            print(env.agent.feedback)

        done = False
        done |= env.get_goal_satisfied()
        done |= env.check_conditions(conditions)
        if done:
            print("You won!")
            break


class TestArgument(argparse.Namespace):
    """Play the abstract text version of an ALFRED environment."""

    """Path to a folder containing PDDL and traj_data files."""
    problem: str

    """Print debug information."""
    debug: bool

    """Load receps."""
    load_receps: bool

    """Path to the reward configuration file."""
    reward_config: str

    """X display to use."""
    x_display: int


def parse_args() -> TestArgument:
    description = "Play the abstract text version of an ALFRED environment."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "problem",
        nargs="?",
        default=None,
        help="Path to a folder containing PDDL and traj_data files."
        f"Default: pick one at random found in {ALFWORLD_DATA}",
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
        default=pjoin(
            next(iter(alfworld.agents.__path__)), "config", "rewards.json"
        ),
        help="Path to the reward configuration file.",
    )
    parser.add_argument("--x-display", default=0, help="X display to use.")
    return parser.parse_args(namespace=TestArgument())


if __name__ == "__main__":
    args = parse_args()

    constants.X_DISPLAY = str(args.x_display)

    if args.problem is None:
        problems = glob.glob(
            pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True
        )
        args.problem = os.path.dirname(random.choice(problems))

    main(args)
