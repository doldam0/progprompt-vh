# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import json
import os
import random
import re
from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from utils.alfworld import ACTIONS, CustomThorEnv, MultipleTaskThorEnv
from utils.types import Graph, TrajectoryData
from utils.utils_aug_env import (add_additional_obj_states,
                                 get_obj_ids_for_adding_states)


class LM:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        max_tokens: int = 128,
        temperature: float = 0,
        stop: Optional[Union[str, List[str]]] = None,
        logprobs: int = 1,
        frequency_penalty: float = 0,
    ):
        os.environ["OPENAI_API_KEY"] = api_key

        self.__model = model
        self.__max_tokens = max_tokens
        self.__temperature = temperature
        self.__stop = stop
        self.__logprobs = logprobs
        self.__frequency_penalty = frequency_penalty

        self.__client = OpenAI()

    def execute(
        self,
        prompt: str,
    ) -> Tuple[ChatCompletion, str]:

        ## function to query LM ##
        # you may adjust the genration parameters as needed
        # more info on parameters here:
        # https://platform.openai.com/docs/api-reference/completions/create
        response = self.__client.chat.completions.create(
            model=self.__model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.__max_tokens,
            temperature=self.__temperature,
            stop=self.__stop,
            logprobs=True,
            top_logprobs=self.__logprobs,
            frequency_penalty=self.__frequency_penalty,
        )

        message = response.choices[0].message.content
        return response, message.strip() if message is not None else ""


def get_current_state_prompt():
    ## fixed function to define "PROMPT for state check"
    current_state_prompt = (
        "kitchencounterdrawer, door is OPEN, character, wallpictureframe, "
        "clothespile is CLOSED, coffeemaker is OFF, pie, wall, bedroom, "
        "microwave is OFF and CLOSED, lightswitch is ON, kitchencabinet is "
        "CLOSED, washingsponge, bellpepper, salmon, fridge is CLOSED, "
        "wallshelf, tvstand, paper, floor, chips, photoframe, kitchen, "
        "whippedcream, candybar, faucet is OFF, tv is OFF, cereal, stovefan, "
        "waterglass, cutleryknife, kitchentable, condimentbottle, wineglass, "
        "bookshelf, cutleryfork, chocolatesyrup, walllamp, bench, sink, "
        "crackers, orchid, condimentshaker, kitchencounter is CLOSED, "
        "livingroom, powersocket, coffeepot is CLOSED, creamybuns, "
        "ceilinglamp, rug, book is CLOSED, plate, toaster is OFF, clock is "
        "OFF, wallphone is OFF, ceiling, fryingpan, box is CLOSED, dishbowl, "
        "bananas, breadslice, bathroom, garbagecan is CLOSED, stove is OFF and "
        "CLOSED, dishwashingliquid, plate ON kitchencounter, cutleryfork ON "
        "kitchentable, bookshelf ON floor, cutleryknife ON kitchentable, "
        "bellpepper ON kitchencounter, microwave ON kitchencounterdrawer, "
        "chocolatesyrup ON wallshelf, whippedcream ON rug, salmon ON "
        "microwave, orchid ON tvstand, wallpictureframe ON wall, bench ON "
        "floor, tvstand ON floor, book INSIDE bookshelf, bananas ON dishbowl, "
        "toaster ON kitchencounterdrawer, whippedcream ON kitchentable, "
        "dishbowl INSIDE bookshelf, fryingpan ON stove, rug ON kitchentable, "
        "coffeepot INSIDE coffeemaker, waterglass ON rug, dishwashingliquid ON "
        "kitchencounter, wallshelf ON wall, washingsponge ON kitchencounter, "
        "clothespile INSIDE bookshelf, bananas INSIDE bookshelf, box ON "
        "bookshelf, plate ON kitchentable, waterglass ON kitchentable, "
        "creamybuns ON wallshelf, breadslice INSIDE toaster, coffeemaker ON "
        "kitchencounterdrawer, chips ON wallshelf, book ON kitchentable, "
        "dishbowl ON bookshelf, pie ON kitchentable, wineglass ON tvstand, box "
        "ON tvstand, coffeepot ON kitchencounter, bellpepper ON "
        "kitchencounterdrawer, condimentshaker INSIDE bookshelf, coffeemaker "
        "ON kitchencounter, toaster ON kitchencounter, box INSIDE bookshelf, "
        "crackers ON wallshelf, character HOLD_RH book, faucet ON "
        "kitchencounter, book ON rug, cereal ON wallshelf, plate INSIDE "
        "microwave, candybar ON wallshelf, condimentbottle INSIDE bookshelf, "
        "tv ON tvstand, microwave ON kitchencounter, paper INSIDE bookshelf, "
        "kitchencounterdrawer ON kitchencounter, fridge ON floor, photoframe "
        "ON tvstand, wallpictureframe ON wallpictureframe, bench ON rug, pie "
        "ON rug, kitchencounterdrawer ON kitchencounterdrawer, dishbowl ON "
        "kitchencounter.\n\n"
        "assert('close' to 'mug' )\n"
        "False\n"
        "assert('close' to 'microwave' )\n"
        "True\n"
        "assert('book' is 'closed' )\n"
        "True\n"
        "assert('lightswitch' is 'OFF')\n"
        "False\n"
        "assert('book' in 'bookshelf')\n"
        "True\n"
        "assert('book' in 'hands')\n"
        "True\n"
        "assert('cereal' on 'bookshelf')\n"
        "False"
    )
    objs = ["microwave", "book", "lightswitch", "bookshelf", "cereal"]
    state, asserts = current_state_prompt = current_state_prompt.split("\n\n")
    state = state.split(",")
    state = "You see: " + ", ".join(
        [i.strip() for i in state if any(element in i for element in objs)]
    )
    current_state_prompt = f"{state}\n\n{asserts}"
    return current_state_prompt


current_state_prompt = get_current_state_prompt()


def run_execution(
    args,
    env: MultipleTaskThorEnv,
    model: LM,
    trajectories: list[tuple[Path, TrajectoryData]],
    tasks: list[str],
    gen_plan: list[str],
    log_file: TextIOWrapper,
) -> Tuple[List[Graph], List[Graph], List[float]]:
    final_states: List[Graph] = []
    initial_states: List[Graph] = []
    exec_per_task: List[float] = []

    task_to_plan = {task: plan for task, plan in zip(tasks, gen_plan)}

    ## initialize and set up enviroenment: visual + graph environment ##
    env.reset(trajectories)
    # TODO: Check if this is needed
    # env.add_character("Chars/Male2", initial_room="kitchen")

    for path, traj_data in (trajectories * 5):
        if any(
            p == path and condition == True
            for p, condition in env.get_which_goal_satisfied()
        ):
            continue

        final_state = {
            "nodes": [],
            "edges": [],
        }

        task_name = traj_data["turk_annotations"]["anns"][0]["task_desc"]
        if task_name not in task_to_plan:
            continue
        plan = task_to_plan[task_name]

        graph = env.environment_graph()
        cc = env.camera_count()
        initial_states.append(graph)

        ## get agent's initial state ##
        agent_has_objid = [
            n["to_id"]
            for n in graph["edges"]
            if n["from_id"] == "agent" and "HOLD" in n["relation_type"]
        ]
        agent_has_obj = [
            n["class_name"]
            for n in graph["nodes"]
            if n["id"] in agent_has_objid
        ]
        # some actions might not execute in the visual simulation, but they will in evolving graphs
        images: list[np.ndarray] = []
        im = env.camera_image()
        images.append(im)
        # s, obj = comm.get_visible_objects(cc-6)
        obj_ids_for_adding_states = get_obj_ids_for_adding_states(graph)
        nodes_with_additional_states = {}

        partial_graph = env.environment_graph(only_visible=True)

        obj_ids_close = [
            n["to_id"]
            for n in graph["edges"]
            if n["from_id"] == "agent" and n["relation_type"] == "CLOSE"
        ]
        obj = [
            node["class_name"]
            for node in partial_graph["nodes"]
            if node["id"] in obj_ids_close
        ]
        obj_ids = {
            node["id"]: node["class_name"]
            for node in graph["nodes"]
            if node["id"] in obj_ids_close and node["class_name"] in obj
        }
        relations = list(
            set(
                obj_ids[n["from_id"]]
                + " "
                + n["relation_type"]
                + " "
                + obj_ids[n["to_id"]]
                for n in graph["edges"]
                if n["from_id"] in obj_ids
                and n["to_id"] in obj_ids
                and n["relation_type"] not in ["CLOSE", "FACING", "INSIDE"]
            )
        )
        obj_states = [
            (node["class_name"], node["states"])
            for node in graph["nodes"]
            if node["class_name"] in obj
        ]
        objs = ""

        for ob_states in obj_states:
            if len(ob_states[1]) > 0:
                objs = (
                    objs
                    + ob_states[0]
                    + " is "
                    + " and ".join(ob_states[1])
                    + ", "
                )
            else:
                objs = objs + ob_states[0] + ", "
        objs = list(set(objs.split(", ")))
        objs = [ob for ob in objs if len(ob) > 0]
        objs = ", ".join(objs) + ", " + ", ".join(relations) + ". "
        if len(agent_has_obj) > 0:
            agent_has_obj = ", ".join(agent_has_obj)
            objs += f" You have {agent_has_obj}. "

        ## parse plan into subgoals ##
        log_file.write(f"\n--Executing task: {traj_data['task_type']}--\n")
        log_file.write(f"Plan:  {plan}\n\n")
        print(f"Executing: {traj_data['task_type']}\n")

        subgoals = {}
        subgoals["0"] = []
        for i in plan.split("\n"):
            i = i.strip()
            if len(i) < 1:
                continue
            if "comments" in args.prompt_task_examples_ablation:
                subgoals["0"].append(i)
            else:
                if "#" in i:
                    sg = i.split("#")[1]
                    sg = sg.strip()
                    subgoals[sg] = []
                else:
                    subgoals[sg].append(i)

        ## begin execution ##
        executable_steps = 0
        total_steps = 0
        last_assert = None
        for subgoal in subgoals.keys():
            step = 1
            for action in subgoals[subgoal]:
                # fixes needed for not getting stuck
                if step > 10:
                    break
                if "grab('wallphone')" in action:
                    continue

                ## state checking ##

                # parse asserts and query LLM
                if "assert" in action:
                    check_state = ""
                    last_assert = action
                    assert_objs = re.findall(r"\b[a-z]+", action)[1::2]
                    state = objs.split(",")
                    state = "You see: " + ", ".join(
                        [
                            i.strip()
                            for i in state
                            if any(ele in i for ele in assert_objs)
                        ]
                    )
                    current_state = (
                        f"{current_state_prompt}\n\n{state}\n\n{action}\n"
                    )
                    _, check_state = model.execute(current_state)
                    log_file.write(
                        f"State check:\n{state}\n{action}\n{check_state.strip()}\n"
                    )
                    continue

                # get recovery actions
                if last_assert != None:
                    if "True" in check_state:
                        # skip revovery if state check is true
                        if "else: " in action:
                            continue
                    elif "False" in check_state:
                        if "else: " in action:
                            action = action.split(": ")[-1].strip()
                        else:
                            state = objs.split(",")
                            state = "You see: " + ", ".join(
                                [
                                    i.strip()
                                    for i in state
                                    if any(ele in i for ele in assert_objs)
                                ]
                            )
                            current_state = f"{current_state_prompt}\n\n{state}\n\n{action}\n"
                            _, check_state = model.execute(current_state)
                            log_file.write(
                                f"State check:\n{state}\n{action}\n{check_state.strip()}\n"
                            )

                # since above steps are not for env, following line go through the env
                total_steps += 1

                ## parse next action
                action = action.lower().split(")")[0]
                action = re.findall(r"\b[a-z]+", action)

                if len(action) == 3 and "put" in action[0]:  # 2 objs action
                    obj_nid1 = [
                        node["id"]
                        for node in graph["nodes"]
                        if action[1] in node["id"]
                        and node["id"] in agent_has_objid
                    ]
                    obj_nid2 = [
                        node["id"]
                        for node in graph["nodes"]
                        if action[2] in node["id"]
                    ]
                    if len(obj_nid1) == 0:
                        step += 1
                        log_file.write("obj not in hand\n")
                        continue
                    if len(obj_nid1) == 1:
                        nid1 = obj_nid1[0]
                    else:
                        nid1 = random.choice(obj_nid1)

                    if len(obj_nid2) == 0:
                        step += 1
                        log_file.write("obj not found\n")
                        continue
                    elif len(obj_nid2) == 1:
                        nid2 = obj_nid2[0]
                    else:
                        nid2 = random.choice(obj_nid2)
                    script_instruction = ACTIONS[action[0]].format(nid1, nid2)
                else:
                    try:
                        nids: list[str] = []
                        err = False
                        for act in action[1:]:
                            ids = [
                                node["id"]
                                for node in graph["nodes"]
                                if act in node["id"]
                            ]
                            if len(ids) == 0:
                                err = True
                                break
                            nids.append(random.choice(ids))
                        if err:
                            step += 1
                            log_file.write("obj not found\n")
                            continue
                        script_instruction = ACTIONS[action[0]].format(*nids)
                    except (KeyError, IndexError) as e:
                        log_file.write(f"bad action: {e}\n")
                        continue

                ## execute next action in both envs: visual and graph
                log_file.write(f"{script_instruction}\n")
                exec_num = env.render_script(script_instruction)
                """
                script = script_instruction[7:]
                try:
                    script = parse_script_line(script, 0)
                except:
                    step += 1
                    continue
                print(script)
                success, final_state, _ = executor.execute(Script([script]))
                """

                # count execution if action executes succesfully in graph env
                executable_steps += exec_num
                # _, graph = comm.environment_graph()
                final_state = env.environment_graph()
                graph = final_state
                partial_graph = env.environment_graph(only_visible=True)
                script_instruction = " ".join(
                    re.findall(r"\b[a-z]+", script_instruction)[1:]
                )
                step += 1

                # get new state info
                agent_has_objid = [
                    n["to_id"]
                    for n in graph["edges"]
                    if n["from_id"] == "agent" and "HOLD" in n["relation_type"]
                ]
                agent_has_obj = [
                    n["class_name"]
                    for n in graph["nodes"]
                    if n["id"] in agent_has_objid
                ]

                # Here you can get an observation, for instance
                im = env.camera_image()
                images.append(im)

                obj_ids_close = [
                    n["to_id"]
                    for n in graph["edges"]
                    if n["from_id"] == "agent" and n["relation_type"] == "CLOSE"
                ]
                obj = [
                    node["class_name"]
                    for node in partial_graph["nodes"]
                    if node["id"] in obj_ids_close
                ]
                obj_ids = dict(
                    [
                        (node["id"], node["class_name"])
                        for node in partial_graph["nodes"]
                        if node["id"] in obj_ids_close
                    ]
                )
                nodes_with_additional_states = add_additional_obj_states(
                    partial_graph,
                    obj_ids_for_adding_states,
                    nodes_with_additional_states,
                )

                relations = list(
                    set(
                        [
                            obj_ids[n["from_id"]]
                            + " "
                            + n["relation_type"]
                            + " "
                            + obj_ids[n["to_id"]]
                            for n in graph["edges"]
                            if n["from_id"] in obj_ids
                            and n["to_id"] in obj_ids
                            and n["relation_type"] not in ["CLOSE", "FACING"]
                        ]
                    )
                )
                obj_states = [
                    (node["class_name"], node["states"])
                    for node in graph["nodes"]
                    if node["class_name"] in obj
                ]
                objs = ""
                for ob_states in obj_states:
                    if len(ob_states[1]) > 0:
                        objs = (
                            objs
                            + ob_states[0]
                            + " is "
                            + " and ".join(ob_states[1])
                            + ", "
                        )
                    else:
                        objs = objs + ob_states[0] + ", "
                objs = list(set(objs.split(", ")))
                objs = [ob for ob in objs if len(ob) > 0]
                objs = ", ".join(objs) + ", " + ", ".join(relations) + ". "

                if len(agent_has_obj) > 0:
                    agent_has_obj = ", ".join(agent_has_obj)
                    objs += f" You have {agent_has_obj}. "

        # augment state with additional state info
        for idx in range(len(final_state["nodes"])):
            if (
                final_state["nodes"][idx]["id"]
                in nodes_with_additional_states.keys()
            ):
                final_state["nodes"][idx] = nodes_with_additional_states[
                    final_state["nodes"][idx]["id"]
                ]

        # get final state for eval
        final_states.append(cast(Graph, final_state))
        exec_per_task.append(executable_steps / total_steps)
    return final_states, initial_states, exec_per_task
