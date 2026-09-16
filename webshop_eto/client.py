"""Agent-R environment client for the ETO / Co-Evolving WebShop protocol.

Drop-in replacement for AgentGym's WebshopEnvClient, exposing exactly the four things Agent-R
touches: conversation_start, reset(idx), observe(), step(action) -> StepOutput. It deliberately
does not import agentenv, so nothing under /data/src/AgentGym is loaded or shadowed.

Two differences from the AgentGym client, both required to match the table:

1. Few-shot. conversation_start carries Co-Evolving's instruction, "OK", and ONE worked example as
   real turns, reproducing prompt_with_icl(..., icl_num=1) with icl_format "conversation"
   (eval_agent/prompt/templates.py). AgentGym's client is zero-shot with a single hardcoded turn.

2. The RAW model reply is sent to the server, not a pre-extracted action. Co-Evolving decides
   between "Observation: Invalid format. The input must contains 'Action: '" and a silent no-op by
   whether `Action:` is present, which the caller cannot express once it has split the string.
"""
import json
import os
from dataclasses import dataclass

import requests

# Uploaded alongside the WebShop source; see k8s/webshop-eto-build.yaml.
ETO_ROOT = os.environ.get("WEBSHOP_ETO_ROOT", "/data/src/webshop-eto")
INSTRUCTION_PATH = f"{ETO_ROOT}/eval_agent/prompt/instructions/webshop_inst.txt"
ICL_PATH = f"{ETO_ROOT}/eval_agent/prompt/icl_examples/webshop_icl.json"
SPLIT_DIR = f"{ETO_ROOT}/eval_agent/data/webshop"


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool


def is_eto():
    """True when this run uses the ETO / Co-Evolving WebShop protocol instead of AgentGym's."""
    return os.environ.get("WEBSHOP_PROTOCOL", "").lower() == "eto"


def replay_conversation_start(conv, env):
    """Seed a fastchat conversation from env.conversation_start.

    The released setup_conversation replays only conversation_start[0] and hardcodes the reply
    "Ok.", which is correct for AgentGym's two-message zero-shot prompt but silently discards the
    in-context example the ETO protocol depends on. Two messages -> released behaviour verbatim;
    more -> replay them all.
    """
    messages = list(env.conversation_start)
    if len(messages) <= 2:
        conv.append_message(conv.roles[0], messages[0]["value"])
        conv.append_message(conv.roles[1], "Ok.")
        return conv
    for msg in messages:
        conv.append_message(conv.roles[0] if msg["from"] == "human" else conv.roles[1], msg["value"])
    return conv


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
    """ETO task ids. 'train' -> train_indices.json (1,824 rows); anything else -> test_indices.json
    (200). eval_agent/tasks/webshop.py maps every non-train split to the test file the same way."""
    name = "train_indices" if split == "train" else "test_indices"
    return json.load(open(f"{SPLIT_DIR}/{name}.json"))


class WebshopEtoEnvClient:
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
        res = requests.post(f"{self.env_server_base}/reset",
                            json={"id": self.env_id, "data_idx": int(data_idx)}, timeout=self.timeout)
        res.raise_for_status()
        self.info = res.json()
        return self.info

    def step(self, action):
        # `action` is the raw assistant reply; the server parses `Action:` out of it.
        res = requests.post(f"{self.env_server_base}/step",
                            json={"id": self.env_id, "action": action}, timeout=self.timeout)
        res.raise_for_status()
        self.info = res.json()
        return StepOutput(state=self.info["observation"], reward=float(self.info["reward"]),
                          done=bool(self.info["done"]))
