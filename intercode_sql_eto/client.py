"""Agent-R environment client for the ETO / Co-Evolving InterCode-SQL protocol.

Drop-in for the AgentGym-style clients Agent-R expects, exposing everything its code touches:
conversation_start, reset(idx), step(action) -> StepOutput, observe(), and .info. It does not import
agentenv, so nothing under /data/src/AgentGym* is loaded or shadowed.

The prompt is rebuilt from Co-Evolving's prompt_with_icl(..., icl_num=1) for
icl_format "conversation": the instruction, "OK", then one worked example - the same construction
webshop_eto and sciworld_eto use, so all three environments present the model with the same shape.
Their intercode_sql_icl.json holds 3 examples of 8 turns; we use the first, as they do.
"""
import json
import os
from dataclasses import dataclass

import requests

ETO_ROOT = os.environ.get("INTERCODE_ETO_ROOT", "/data/src/intercode-sql-eto")
INSTRUCTION_PATH = f"{ETO_ROOT}/eval_agent/prompt/instructions/intercode_sql_inst.txt"
ICL_PATH = f"{ETO_ROOT}/eval_agent/prompt/icl_examples/intercode_sql_icl.json"
DATA = f"{ETO_ROOT}/eval_agent/data/intercode_sql"


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool


def is_eto():
    return os.environ.get("INTERCODE_PROTOCOL", "").lower() == "eto"


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


def stable_unique(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_split(split):
    """Positions in this list are what reset() takes; the server resolves them to spider records."""
    name = "train" if split == "train" else "test"
    return stable_unique(json.load(open(f"{DATA}/{name}_indices.json")))


class IntercodeSqlEtoEnvClient:
    conversation_start = build_conversation_start(icl_num=1)

    def __init__(self, env_server_base, data_len=200, timeout=600):
        self.env_server_base = env_server_base.rstrip("/")
        self.data_len = data_len
        self.timeout = timeout
        self.info = {"observation": "", "reward": 0.0, "score": 0.0, "done": False}
        res = requests.post(f"{self.env_server_base}/create", json={}, timeout=self.timeout)
        res.raise_for_status()
        self.env_id = res.json()["id"]

    def __len__(self):
        return self.data_len

    def observe(self):
        return self.info["observation"]

    def reset(self, data_idx=0):
        r = requests.post(f"{self.env_server_base}/reset",
                          json={"id": self.env_id, "data_idx": int(data_idx)}, timeout=self.timeout)
        r.raise_for_status()
        self.info = r.json()
        return self.info

    def step(self, action):
        # The raw reply goes to the server, which runs Co-Evolving's parse_sql_action on it: the
        # action is a fenced SQL block, so splitting on "Action:" here would break the fence.
        r = requests.post(f"{self.env_server_base}/step",
                          json={"id": self.env_id, "action": action}, timeout=self.timeout)
        r.raise_for_status()
        self.info = r.json()
        return StepOutput(state=self.info["observation"], reward=float(self.info["reward"]),
                          done=bool(self.info["done"]))
