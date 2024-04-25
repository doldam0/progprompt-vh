# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
This script evaluates plan generation using openAI LLMs
for the VirtualHome environment tasks
"""
import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))

import argparse
import json
import random
import time
from typing import Dict, List, TypedDict

import openai

from utils.alfworld import CustomThorEnv
from utils.loads import (
    load_annotations,
    load_environment_states,
    load_plan,
    load_trajectories,
)
from utils.utils_execute import *


class RunEvalArguments(argparse.Namespace):
    data_root: str
    progprompt_path: str
    expt_name: str
    openai_api_key: str
    display: str
    screen_height: int
    screen_width: int
    gpt_version: str
    test_set: str
    prompt_task_examples: str
    seed: int
    prompt_num_examples: int
    prompt_task_examples_ablation: str
    load_generated_plans: bool


def parse_args() -> RunEvalArguments:
    parser = argparse.ArgumentParser()

    parser.add_argument("data_root", type=str)
    parser.add_argument("--progprompt-path", type=str, required=True)
    parser.add_argument("--expt-name", type=str, required=True)

    parser.add_argument("--openai-api-key", type=str, default="sk-xyz")
    parser.add_argument("--display", type=str, default="0")
    parser.add_argument("--screen-height", type=int, default=300)
    parser.add_argument("--screen-width", type=int, default=300)

    parser.add_argument(
        "--gpt-version",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "davinci", "curie"],
    )

    parser.add_argument(
        "--test-set",
        type=str,
        default="env1",
        choices=[
            "test_unseen",
            "test_seen",
            "test_unseen_ambiguous",
            "env1",
            "env2",
        ],
    )
    parser.add_argument(
        "--prompt-task-examples",
        type=str,
        default="default",
        choices=["default", "random"],
    )
    # for random task examples, choose seed
    parser.add_argument("--seed", type=int, default=0)

    ## NOTE: davinci or older GPT3 versions have a lower token length limit
    ## check token length limit for models to set prompt size:
    ## https://platform.openai.com/docs/models
    parser.add_argument(
        "--prompt-num-examples", type=int, default=3, choices=range(1, 7)
    )
    parser.add_argument(
        "--prompt-task-examples-ablation",
        type=str,
        default="none",
        choices=["none", "no_comments", "no_feedback", "no_comments_feedback"],
    )

    parser.add_argument("--load-generated-plans", type=bool, default=False)

    args = parser.parse_args(namespace=RunEvalArguments())
    return args


class EvaluationResult(TypedDict):
    PSR: float
    SR: float
    Precision: float
    Exec: float


def eval(
    final_states: List[Graph],
    final_states_GT: List[Graph],
    initial_states: List[Graph],
    test_tasks: List[str],
    exec_per_task: List[float],
    log_file: TextIOWrapper,
) -> Dict[str, EvaluationResult]:

    ## the evaluation criteria is not perfect
    ## since sometimes the tasks are underspecified, like which object to interact with
    ## for example "turn off lightswitch" could happen in multiple locations
    ## the evaluation happens w.r.t one possible valid state
    ## that the annotator provides

    sr: List[float] = []
    unsatif_conds: List[int] = []
    unchanged_conds: List[int] = []
    total_goal_conds: List[int] = []
    total_unchanged_conds: List[int] = []
    results: Dict[str, EvaluationResult] = {}
    for g, g_gt, g_in, d in zip(
        final_states, final_states_GT, initial_states, test_tasks
    ):
        obj_ids = dict(
            [(node["id"], node["class_name"]) for node in g_in["nodes"]]
        )
        relations_in = set(
            [
                obj_ids[n["from_id"]]
                + " "
                + n["relation_type"]
                + " "
                + obj_ids[n["to_id"]]
                for n in g_in["edges"]
            ]
        )
        obj_states_in = set(
            [
                node["class_name"] + " " + st
                for node in g_in["nodes"]
                for st in node["states"]
            ]
        )

        obj_ids = dict(
            [(node["id"], node["class_name"]) for node in g["nodes"]]
        )
        relations = set(
            [
                obj_ids[n["from_id"]]
                + " "
                + n["relation_type"]
                + " "
                + obj_ids[n["to_id"]]
                for n in g["edges"]
            ]
        )
        obj_states = set(
            [
                node["class_name"] + " " + st
                for node in g["nodes"]
                for st in node["states"]
            ]
        )

        obj_ids = dict(
            [(node["id"], node["class_name"]) for node in g_gt["nodes"]]
        )
        relations_gt = set(
            [
                obj_ids[n["from_id"]]
                + " "
                + n["relation_type"]
                + " "
                + obj_ids[n["to_id"]]
                for n in g_gt["edges"]
            ]
        )
        obj_states_gt = set(
            [
                node["class_name"] + " " + st
                for node in g_gt["nodes"]
                for st in node["states"]
            ]
        )

        log_file.write(
            f"\nunsatisfied state conditions: relations: {(relations_gt - relations_in) - (relations - relations_in)}, object states: {(obj_states_gt - obj_states_in) - (obj_states - obj_states_in)}"
        )
        unsatif_conds.append(
            (
                len((relations_gt - relations_in) - (relations - relations_in))
                + len(
                    (obj_states_gt - obj_states_in)
                    - (obj_states - obj_states_in)
                )
            )
        )
        total_goal_conds.append(
            len(relations_gt - relations_in)
            + len(obj_states_gt - obj_states_in)
        )
        sr.append(1 - unsatif_conds[-1] / total_goal_conds[-1])

        unchanged_conds.append(
            (
                len(relations_gt.intersection(relations_in) - relations)
                + len(obj_states_gt.intersection(obj_states_in) - obj_states)
            )
        )
        total_unchanged_conds.append(
            len(relations_gt.intersection(relations_in))
            + len(obj_states_gt.intersection(obj_states_in))
        )

        results[d] = {
            "PSR": sr[-1],
            "SR": sr[-1:].count(1.0),
            "Precision": 1 - unchanged_conds[-1] / total_unchanged_conds[-1],
            "Exec": exec_per_task[-1],
        }

    results["overall"] = {
        "PSR": sum(sr) / len(sr),
        "SR": sr.count(1.0) / len(sr),
        "Precision": 1 - sum(unchanged_conds) / sum(total_unchanged_conds),
        "Exec": sum(exec_per_task) / len(exec_per_task),
    }
    return results


def planner_executer(args: RunEvalArguments):

    # initialize env
    env = CustomThorEnv(
        x_display=args.display,
        player_screen_height=args.screen_height,
        player_screen_width=args.screen_width,
    )

    trajs = load_trajectories(args.data_root)

    # prompt example environment is set to env_id 0
    env.reset(*trajs[0])

    env_graph = env.environment_graph()
    obj = list(set([node["class_name"] for node in env_graph["nodes"]]))

    # define available actions and append avaailable objects from the env
    prompt = f"from actions import turnright, turnleft, walkforward, walktowards <obj>, walk <obj>, run <obj>, grab <obj>, switchon <obj>, switchoff <obj>, open <obj>, close <obj>, lookat <obj>, sit <obj>, standup, find <obj>, turnto <obj>, drink <obj>, pointat <obj>, watch <obj>, putin <obj> <obj>, putback <obj> <obj>"
    prompt += f"\n\nobjects = {obj}"

    # load train split for task examples
    tmp = load_plan(
        f"{args.progprompt_path}/data/pythonic_plans/train_complete_plan_set.json"
    )
    prompt_egs = {}
    for k, v in tmp.items():
        prompt_egs[k] = v
    # print("Loaded %d task example" % len(prompt_egs.keys()))

    ## define the prompt example task setting ##

    # default examples from the paper
    if args.prompt_task_examples == "default":
        default_examples = [
            "put_the_wine_glass_in_the_kitchen_cabinet",
            "throw_away_the_lime",
            "wash_mug",
            "refrigerate_the_salmon",
            "bring_me_some_fruit",
            "wash_clothes",
            "put_apple_in_fridge",
        ]
        for i in range(args.prompt_num_examples):
            prompt += "\n\n" + prompt_egs[default_examples[i]]

    # random egs - change seeds
    if args.prompt_task_examples == "random":
        random.seed(args.seed)
        prompt_egs_keys = random.sample(
            list(prompt_egs.keys()), args.prompt_num_examples
        )

        for eg in prompt_egs_keys:
            prompt += "\n\n" + prompt_egs[eg]

    # abalation settings
    if args.prompt_task_examples_ablation == "no_comments":
        prompt = prompt.split("\n")
        prompt = [line for line in prompt if "# " not in line]
        prompt = "\n".join(prompt)

    if args.prompt_task_examples_ablation == "no_feedback":
        prompt = prompt.split("\n")
        prompt = [
            line
            for line in prompt
            if not any([x in line for x in ["assert", "else"]])
        ]
        prompt = "\n".join(prompt)

    if args.prompt_task_examples_ablation == "no_comments_feedback":
        prompt = prompt.split("\n")
        prompt = [
            line
            for line in prompt
            if not any([x in line for x in ["assert", "else", "# "]])
        ]
        prompt = "\n".join(prompt)

    # setup logging
    log_filename = f"{args.expt_name}_{args.prompt_task_examples}_{args.prompt_num_examples}examples"
    if args.prompt_task_examples_ablation != "none":
        log_filename += f"_{args.prompt_task_examples_ablation}"
    log_filename += f"_{args.test_set}"
    log_file = open(
        f"{args.progprompt_path}/results/{log_filename}_logs.txt", "w"
    )
    log_file.write(f"\n----PROMPT for planning----\n{prompt}\n")

    test_tasks: List[str] = []

    # evaluate in given unseen env
    env.reset(*trajs[0])
    graph = env.environment_graph()
    obj = list(set([node["class_name"] for node in graph["nodes"]]))
    prompt += f"\n\n\nobjects = {obj}"

    # evaluation tasks in given unseen env
    test_tasks = [
        list(annotation.keys())[0]
        for annotation in load_annotations(
            f"{args.progprompt_path}/data/new_env/{args.test_set}_annotated.json"
        )
    ]
    log_file.write(
        f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n"
    )

    # test_tasks = test_tasks[:3] ## debug to check changes

    planning_model = LM(
        args.gpt_version,
        api_key=args.openai_api_key,
        max_tokens=600,
        stop=["def"],
        frequency_penalty=0.15,
    )

    executing_model = LM(
        args.gpt_version,
        api_key=args.openai_api_key,
        max_tokens=2,
        stop=["\n"],
    )

    # generate plans for the test set
    if not args.load_generated_plans:
        gen_plan: List[str] = []
        for task in test_tasks:
            print(f"Generating plan for: {task}\n")
            prompt_task = "def {fxn}():".format(fxn="_".join(task.split(" ")))
            curr_prompt = f"{prompt}\n\n{prompt_task}\n\t"
            _, text = planning_model.execute(curr_prompt)
            gen_plan.append(text)
            # because codex has query limit per min
            if args.gpt_version == "code-davinci-002":
                time.sleep(90)

        # save generated plan
        line: Dict[str, str] = {}
        print(f"Saving generated plan at: {log_filename}_plans.json\n")
        with open(
            f"{args.progprompt_path}/results/{log_filename}_plans.json", "w"
        ) as f:
            for plan, task in zip(gen_plan, test_tasks):
                line[task] = plan
            json.dump(line, f)

    # load from file
    else:
        print(f"Loading generated plan from: {log_filename}.json\n")
        data = load_plan(
            f"{args.progprompt_path}/results/{log_filename}_plans.json"
        )
        test_tasks, gen_plan = (list(e) for e in zip(*data.items()))

    log_file.write(
        f"\n----PROMPT for state check----\n{current_state_prompt}\n"
    )

    # run execution
    print(f"\n----Runing execution----\n")
    final_states, initial_states, exec_per_task = run_execution(
        args, env, executing_model, trajs, test_tasks, gen_plan, log_file
    )

    # evaluate
    final_states_GT = list(
        load_environment_states(
            f"{args.progprompt_path}/data/final_states/final_states_{args.test_set}.json"
        )
    )

    results = eval(
        final_states,
        final_states_GT,
        initial_states,
        test_tasks,
        exec_per_task,
        log_file,
    )

    print(f"\n----Results----\n{results['overall']}\n")
    with open(
        f"{args.progprompt_path}/results/{log_filename}_metric.json", "w"
    ) as f:
        json.dump(results, f)
    log_file.close()


if __name__ == "__main__":
    args = parse_args()
    openai.api_key = args.openai_api_key

    if not osp.isdir(f"{args.progprompt_path}/results/"):
        os.makedirs(f"{args.progprompt_path}/results/")

    planner_executer(args=args)
