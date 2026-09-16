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
from fastchat.model.model_adapter import get_conversation_template
from mcts_utils.llm_server import *
from agentenv.envs import WebshopEnvClient, SciworldEnvClient, TextCraftEnvClient
import argparse
import os

Task = os.environ["TASK"]

if Task == "webshop":
    from mcts_utils.webshop.mcts_ws import *
elif Task == "sciworld":
    from mcts_utils.sciworld.mcts_sci import *
elif Task == "textcraft":
    from mcts_utils.textcraft.mcts_tc import *

def initialize_environment(Task: str, env_server_base: str, data_len: int = 200):
    """
    Initializes the appropriate environment based on the task type.
    """
    if Task == "webshop":
        return WebshopEnvClient(env_server_base=env_server_base, data_len=data_len)
    elif Task == "sciworld":
        return SciworldEnvClient(env_server_base=env_server_base, data_len=data_len)
    elif Task == "textcraft":
        return TextCraftEnvClient(env_server_base=env_server_base, data_len=data_len)
    else:
        raise ValueError(f"Unknown Task: {Task}")

def setup_conversation(env):
    """
    Sets up the initial conversation for the environment.
    """
    conversation = list(env.conversation_start)
    conv = get_conversation_template('gpt-4')

    conv.append_message(conv.roles[0], conversation[0]["value"])
    conv.append_message(conv.roles[1], 'Ok.')
    observation = env.observe() if os.environ["TASK"] == "webshop" else env.info["observation"]
    conv.append_message(conv.roles[0], observation)
    return conv

def main(Task: str, model_name: str, env_server_base: str, max_steps: int):
    """
    Main execution function for handling tasks and initiating tests.
    """
    # Initialize environment
    env = initialize_environment(Task, env_server_base)

    # Load task indices (test_id/ is not shipped; the ids live under mcts_utils/<task>/)
    test_file = f"test_id/{Task}_test.json"
    if not os.path.exists(test_file):
        test_file = f"mcts_utils/{Task}/{Task}_test.json"
    temp = read_json(test_file)
    task_inds = [ind["item_id"].replace(f"{Task}_", "") for ind in temp]
    if "TASK_LIMIT" in os.environ:
        task_inds = task_inds[:int(os.environ["TASK_LIMIT"])]

    # Eval is sharded over GPUs: shard i takes every TASK_SHARDS-th id. Every shard writes to the
    # same result directory and the skip check below ignores finished ids, so shards never collide.
    shards = int(os.environ.get("TASK_SHARDS", 1))
    if shards > 1:
        task_inds = task_inds[int(os.environ["TASK_SHARD"])::shards]

    # Load the model once; constructing FuncCallOffline per task re-creates the vLLM engine
    calling = FuncCallOffline(model_name=model_name)

    # Process each task index
    for idx in task_inds:
        # Must match the output directory used by perform_test
        dir_path = f"test_result/{Task}/{model_name}_{os.environ['MODEL_TYPE']}"
        file_path = f"{dir_path}/search_results_{idx}.json"

        if os.path.exists(file_path):
            print(f"{file_path} exists. Skipping.")
            continue

        env.reset(int(idx))
        conv = setup_conversation(env)
        perform_test(calling, env, conv, model_name, idx, max_steps)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run MCTS tests for specified tasks.")
    parser.add_argument("--env_server_base", type=str, default="http://127.0.0.1:8000", help="Base URL for the environment server.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06", help="Model name to be used.")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps allowed for a task.")
    args = parser.parse_args()

    # Load environment variables
    Task = os.environ.get("TASK")
    if not Task:
        raise ValueError("The TASK environment variable is not set.")

    # Execute main function
    main(Task, args.model_name, args.env_server_base, args.max_steps)