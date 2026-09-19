"""
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""
import json
from copy import deepcopy
import os
from tqdm import tqdm
import random
import argparse
from mcts_utils.llm_server import *

Task = os.environ["TASK"]

if Task == "webshop":
    from mcts_utils.webshop.mcts_ws import *
elif Task == "sciworld":
    from mcts_utils.sciworld.mcts_sci import *
elif Task == "textcraft":
    from mcts_utils.textcraft.mcts_tc import *


def revise_worst_path(calling, worst_path, best_path, task_description):
    """
    Revises the worst path by evaluating each action using a verifier and provides feedback.

    Args:
        calling: An instance for making calls (e.g., LLM function).
        worst_path: Path representing a sequence of bad nodes.
        best_path: Path representing a sequence of good nodes.
        task_description: Description of the task being solved.

    Returns:
        tuple: A revised path, truncated best path, and a list of revision feedback.
    """
    revise_feedback = []
    action_obs = []
    revise_path = [worst_path[0]]
    flag = 1
    node = worst_path[0]

    # Initialize action-observation pairs for the initial node
    Task = os.environ["TASK"]
    if Task == "sciworld":
        for recent_action in node.recent_actions:
            act, obs = map(str.strip, recent_action.replace("(", "").replace(")", "").split(","))
            action_obs.append(f"Action: {act}")
            action_obs.append(f"Observation: {obs}")
    else:
        for recent_action in node.recent_actions:
            action_obs.append(f"Action: {recent_action[0]}")
            action_obs.append(f"Observation: {recent_action[1]}")

    # Process each node in the worst path
    for node in worst_path[1:]:
        # Generate the prompt for the verifier
        action_obs_prompt = '\n'.join(action_obs)
        max_len = int(os.environ["MAX_TOKEN_LENGTH"])
        while len(action_obs_prompt.split()) > max_len - 60:
            action_obs_prompt = action_obs_prompt[6:]  # Truncate prompt if too long

        prompt = prompt_template.format(
            task_description=task_description,
            action_obs_prompt=action_obs_prompt,
            action=node.action,
            observation=node.obs
        )

        # Call the LLM to evaluate the action
        message = [{"role": "user", "content": prompt}]
        response = calling.llm_func(message, os.environ["MODEL_NAME"])

        # Extract the judgement and feedback
        judgement = response.split('Judgement:')[-1].strip()
        revise_feedback.append({
            "revision_feedback": response,
            "action": node.action,
            "obs": node.obs,
            "judgement": judgement
        })

        # Update the revised path based on the judgement
        if "good" in judgement.lower() or "uncertain" in judgement.lower():
            revise_path.append(node)
        else:
            if node.disaster:  # Stop revision if the node represents a disaster
                flag = 0
            revise_path.append(node)
            break

        # Update the action-observation log for the next iteration
        action_obs.append(f"Action: {node.action}")
        action_obs.append(f"Observation: {node.obs}")

    return revise_path, best_path[1:], revise_feedback


def node_avg_value(node):
    return node.value / node.visits if node.visits > 0 else 0.0

def find_leaf_paths(node):
    def dfs(current_node, path):
        if current_node.is_terminal and not current_node.children:
            leaf_paths.append(path + [current_node])
            return
        for child in current_node.children:
            dfs(child, path + [current_node])

    leaf_paths = []
    dfs(node, [])
    return leaf_paths

def sort_leaf_paths_by_value(leaf_paths):
    return sorted(leaf_paths, key=lambda path: node_avg_value(path[-1]), reverse=True)

def pair_leaf_paths(leaf_paths):

    random.seed(123)
    paired_paths = [
        (leaf_paths[i], leaf_paths[j])
        for i in range(len(leaf_paths))
        for j in range(len(leaf_paths))
        if i != j and node_avg_value(leaf_paths[i][-1]) - node_avg_value(leaf_paths[j][-1]) > float(os.environ["BETA"])
    ]
    random.shuffle(paired_paths)
    return paired_paths

def conversation_generation(bad_node_path, good_node_path, revision_thought=None):

    # Initialize the conversation log
    log_temp = [conv for conv in bad_node_path[0].state]

    # Generate the bad path log
    bad_log = []
    for node in bad_node_path[1:]:
        bad_log.append({"role": "assistant", "content": node.llm_response, "loss": False})
        bad_log.append({"role": "user", "content": node.obs})

    # Append a revision thought to the bad log. When pair sharding pre-draws the sentence (see
    # main), it is passed in; the released behaviour is the draw below.
    if revision_thought is None:
        revision_thought = revision_thoughts[random.randint(0, 9)]
    bad_log.append({"role": "assistant", "content": f"Thought: {revision_thought}\nAction: wait", "loss": True})
    bad_log.append({"role": "user", "content": "ok."})

    # Generate the good path log
    good_log = []
    k, flag = 0, 0
    for node in good_node_path:
        if flag == 0 and k < len(bad_log) and bad_log[k]["content"] == node.llm_response:
            k += 2
            continue
        flag = 1
        good_log.append({"role": "assistant", "content": node.llm_response, "loss": True})
        good_log.append({"role": "user", "content": node.obs})

    return log_temp + bad_log + good_log

def conversation_generation_good(good_node_path):

    log_temp = [conv for conv in good_node_path[0].state]

    for node in good_node_path[1:]:
        log_temp.append({"role": "assistant", "content": node.llm_response, "loss": True})
        log_temp.append({"role": "user", "content": node.obs})

    return log_temp

def main(calling, data_path, output_dir, task_num, data_type, revise=False, pair_shard=0, pair_shards=1):
    """
    Processes a task using the ExtendedMCTS tree structure and generates logs.

    Args:
        calling: An instance for making calls (e.g., FuncCallOffline).
        data_path: Path to the input data file.
        output_dir: Directory to save the processed output.
        task_num: Task identifier.
        data_type: Type of data to process ("good" or other types).
        revise: Boolean indicating whether to apply revision logic.
    """
    # Load the MCTS root node
    root = ExtendedMCTS.load(data_path)
    if root.value <= 0:
        return

    # Find and sort leaf paths
    leaf_paths = find_leaf_paths(deepcopy(root))
    sorted_leaf_paths = sort_leaf_paths_by_value(leaf_paths)

    # Handle "good" data type (independent of pairs; under pair sharding only shard 0 writes it)
    if data_type == 'good':
        if pair_shard != 0:
            return
        for high_path in sorted_leaf_paths:
            if float(high_path[-1].value) <= float(os.environ["ALPHA"]):
                continue
            revise_log = conversation_generation_good(high_path)
            output_entry = {
                "task_num": task_num,
                "revise_log": revise_log,
                "revise": revise
            }
            write_to_jsonl(output_dir, output_entry)
        return

    # Pair paths and process them
    paired_paths = pair_leaf_paths(sorted_leaf_paths)

    # Pair sharding (pair_shards=1 is the released, serial behaviour).
    # Pre-draw the revision sentence for every alpha-passing pair, in serial order, from the RNG
    # state pair_leaf_paths leaves behind. The released loop consumes exactly one randint per pair
    # that reaches conversation_generation, in that order, and nothing else in this file or in the
    # LLM client touches `random` - so every shard reproduces the serial assignment exactly.
    # Shard i then processes pairs i, i+N, i+2N, ... The per-pair work below is unchanged.
    alpha_env = float(os.environ["ALPHA"])
    thought_for = {}
    for i, (hp, _) in enumerate(paired_paths):
        if float(hp[-1].value) > alpha_env:
            thought_for[i] = revision_thoughts[random.randint(0, 9)]
    my_pairs = [(i, p) for i, p in enumerate(paired_paths) if i % pair_shards == pair_shard]
    k_ind = 0

    for i, (high_path, low_path) in tqdm(my_pairs, desc="Processing pairs"):
        k_ind += 1
        if float(high_path[-1].value) <= float(os.environ["ALPHA"]):
            continue

        # Generate conversation logs
        high_log = conversation_generation_good(high_path)
        low_log = conversation_generation_good(low_path)
        task_description = low_path[-1].state[3]["content"]

        if revise:
            bad_node_path, good_node_path, revise_feedback = revise_worst_path(
                calling, low_path, high_path, task_description
            )
            revise_log = conversation_generation(bad_node_path, good_node_path, thought_for[i])
        else:
            revise_feedback = ""
            bad_node_path, good_node_path = low_path, high_path[1:]
            revise_log = conversation_generation(bad_node_path, good_node_path, thought_for[i])

        # Write the results to a JSONL file
        output_entry = {
            "task_num": task_num,
            "revise_log": revise_log,
            "high_log": high_log,
            "low_log": low_log,
            "task_description": task_description,
            "revise": revise,
            "revise_feedback": revise_feedback
        }
        write_to_jsonl(output_dir, output_entry)

        

def process_files(calling, input_files, input_dir, output_dir, data_type, revise, pair_shard=0, pair_shards=1):
    """
    Processes a list of input files and calls the main function for each file.

    Args:
        calling: An instance for making calls (e.g., FuncCallOffline).
        input_files: List of input file names.
        input_dir: Directory containing the input files.
        output_dir: Directory to save the processed output.
        data_type: Type of data being processed (e.g., "centric").
        revise: Boolean indicating whether to revise data.
    """
    for input_file in tqdm(input_files, desc="Processing files"):
        task_num = input_file.split('_')[-1].split('.')[0]
        main(
            calling,
            os.path.join(input_dir, input_file),
            output_dir,
            task_num,
            data_type,
            revise=revise,
            pair_shard=pair_shard,
            pair_shards=pair_shards,
        )

def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process MCTS training data.")
    parser.add_argument("--data_type", type=str, default="centric", help="Type of data to process (default: centric).")
    parser.add_argument("--revise", type=int, default=1, help="Whether to revise the data (1 for True, 0 for False).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files.")
    parser.add_argument("--output_dir", type=str, default="mcts_training_data", help="Directory for output files.")
    parser.add_argument("--max", type=int, default=200, help="Maximum range for data processing (default: 200).")
    parser.add_argument("--min", type=int, default=0, help="Minimum range for data processing (default: 0).")
    parser.add_argument("--step", type=int, default=10, help="Step size for data processing (default: 10).")
    parser.add_argument("--pair_shard", type=int, default=0,
                        help="Pair sharding: this process handles pairs whose index modulo pair_shards equals this (default 0).")
    parser.add_argument("--pair_shards", type=int, default=1,
                        help="Pair sharding: total number of shards (default 1 = the released serial behaviour).")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Initialize calling object
    calling = None
    if args.data_type == "centric":
        calling = FuncCallOffline(model_name=os.environ["MODEL_NAME"])

    # Set parameters
    data_type = args.data_type
    revise = bool(args.revise)
    task_name = os.environ.get("TASK", "default_task")
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Prepare input files
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    sorted_files = sorted(input_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    output_file = os.path.join(output_dir, f"{task_name}_{data_type}.jsonl")

    # Process files
    process_files(calling, sorted_files, input_dir, output_file, data_type, revise,
                  pair_shard=args.pair_shard, pair_shards=args.pair_shards)