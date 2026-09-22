"""Agent-R environment client for the ETO / Co-Evolving ScienceWorld protocol.

Drop-in for AgentGym's SciworldEnvClient, exposing everything Agent-R's SciWorld path touches:
conversation_start, reset(idx), step(action) -> StepOutput, observe(), .info, and the three lookups
findValidActionNew needs (get_look_around, get_inventory,
get_valid_action_object_combinations). It does not import agentenv, so nothing under
/data/src/AgentGym-agentr is loaded or shadowed.

Differences from the AgentGym client, all required to match the table:
  * few-shot prompt rebuilt from their prompt_with_icl(..., icl_num=1) - instruction, "OK", then one
    14-turn worked example, as eval_agent/configs/task/sciworld.json specifies
    (icl_format: conversation). AgentGym's client is zero-shot with one hardcoded turn.
  * tasks addressed by position in their (task_name, variation_idx) split files, resolved server-side.
  * max_steps_for() exposes the PER-TASK budget from max_steps.json (10-120); AgentGym and Agent-R
    both assume a single global number.
"""
import json
import os
from dataclasses import dataclass

import requests

ETO_ROOT = os.environ.get("SCIWORLD_ETO_ROOT", "/data/src/sciworld-eto")
INSTRUCTION_PATH = f"{ETO_ROOT}/eval_agent/prompt/instructions/sciworld_inst.txt"
ICL_PATH = f"{ETO_ROOT}/eval_agent/prompt/icl_examples/sciworld_icl.json"
DATA = f"{ETO_ROOT}/eval_agent/data/sciworld"


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool


def is_eto():
    return os.environ.get("SCIWORLD_PROTOCOL", "").lower() == "eto"


def build_conversation_start(icl_num=1):
    """Reproduce eval_agent/prompt/templates.py prompt_with_icl for icl_format 'conversation'."""
    instruction = open(INSTRUCTION_PATH).read().rstrip("\n")
    raw_icl = json.load(open(ICL_PATH))
    messages = [{"from": "human", "loss": None, "value": instruction}]
    for i in range(min(icl_num, len(raw_icl))):
        for j, turn in enumerate(raw_icl[i]):
            content = turn["content"]
            if i == 0 and j == 0:
                messages.append({"from": "gpt", "loss": False, "value": "OK"})
                messages.append({"from": "human", "loss": None, "value": content})
                continue
            role = "human" if j % 2 == 0 else "gpt"
            messages.append({"from": role, "loss": None if role == "human" else False, "value": content})
    return tuple(messages)


def load_split(split):
    name = "train" if split == "train" else ("dev" if split == "dev" else "test")
    return [tuple(x) for x in json.load(open(f"{DATA}/{name}_indices.json"))]


def train_order():
    """Train-split positions, round-robin over task type.

    Agent-R's released SciWorld collection walks task_nums[min:max] and takes variations 1..9 of
    EACH task, i.e. 23 task types x 9 = 207 simulations spread uniformly over the task types. ETO's
    train_indices.json is instead grouped by task, so a contiguous [0, 200) prefix is badly skewed:
    task-2a-test-conductivity alone would be 42 of the 200 while 4 of the 24 types get none.

    Interleaving by type restores the released code's shape on ETO's addressing - the first 200
    positions become 8-9 variations of every one of the 24 types - while the values are still
    ETO split positions, which is what env.reset() expects.
    """
    tasks = load_split("train")
    by_type = {}
    for pos, (task_name, _) in enumerate(tasks):
        by_type.setdefault(task_name, []).append(pos)
    order = []
    for k in range(max(len(v) for v in by_type.values())):
        for positions in by_type.values():        # dict preserves first-seen task order
            if k < len(positions):
                order.append(positions[k])
    return order


class SciworldEtoEnvClient:
    conversation_start = build_conversation_start(icl_num=1)

    def __init__(self, env_server_base, data_len=200, timeout=600):
        self.env_server_base = env_server_base.rstrip("/")
        self.data_len = data_len
        self.timeout = timeout
        self.info = {"observation": "", "reward": 0.0, "score": 0.0, "done": False}
        self._max_steps = json.load(open(f"{DATA}/max_steps.json"))
        res = requests.post(f"{self.env_server_base}/create", json={}, timeout=self.timeout)
        res.raise_for_status()
        self.env_id = res.json()["id"]

    def __len__(self):
        return self.data_len

    def _get(self, path):
        r = requests.get(f"{self.env_server_base}/{path}?env_idx={self.env_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def observe(self):
        return self.info["observation"]

    def reset(self, data_idx=0):
        r = requests.post(f"{self.env_server_base}/reset",
                          json={"id": self.env_id, "data_idx": int(data_idx)}, timeout=self.timeout)
        r.raise_for_status()
        self.info = r.json()
        return self.info

    def step(self, action):
        r = requests.post(f"{self.env_server_base}/step",
                          json={"id": self.env_id, "action": action}, timeout=self.timeout)
        r.raise_for_status()
        self.info = r.json()
        return StepOutput(state=self.info["observation"], reward=float(self.info["reward"]),
                          done=bool(self.info["done"]))

    # --- what findValidActionNew needs ---
    def get_look_around(self):
        return self._get("look_around").get("look_around", "")

    def get_inventory(self):
        return self._get("inventory").get("inventory", "")

    def get_valid_action_object_combinations(self):
        return self._get("valid_action_object_combinations").get("get_valid_action_object_combinations", "")

    def max_steps_for(self, data_idx, split="test"):
        """Co-Evolving's per-task budget for this task (10-120), not one global number."""
        tasks = load_split(split)
        task_name, _ = tasks[int(data_idx) % len(tasks)]
        return self._max_steps.get(task_name)
