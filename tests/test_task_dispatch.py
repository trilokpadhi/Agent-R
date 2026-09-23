"""Every Task dispatch point resolves, without a cluster or a GPU.

    python3 tests/test_task_dispatch.py

Adding an environment means touching several `if Task == ...` chains across mcts_collection.py and
eval.py. Missing one is invisible until a job dies minutes in: the InterCode-SQL smoke run failed
twice that way - once on an unconditional `agentenv` import, once on initialize_environment having
no branch for the new task. Both would have been caught here in a second.

The network and the packages only the policy image carries are stubbed, so this runs anywhere.
It needs the Co-Evolving checkout for the prompt and split files, because the eto clients read
those at import time (set COEVOLVING_ROOT to point elsewhere).
"""
import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CO = os.environ.get("COEVOLVING_ROOT", str(ROOT / "Co-Evolving-Agents"))

CASES = [
    ("intercode_sql", {"INTERCODE_PROTOCOL": "eto"}, "IntercodeSqlEtoEnvClient", 10),
    ("webshop",       {"WEBSHOP_PROTOCOL": "eto"},   "WebshopEtoEnvClient",      None),
    ("sciworld",      {"SCIWORLD_PROTOCOL": "eto"},  "SciworldEtoEnvClient",     16),
]


# ---------------------------------------------------------------- child
class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _install_stubs():
    def post(url, json=None, timeout=None):
        if url.endswith("/create"):
            return _Resp({"id": 0})
        if url.endswith("/reset"):
            return _Resp({"observation": "a task", "reward": 0.0, "score": 0.0,
                          "done": False, "max_steps": 20})
        return _Resp({"observation": "an observation", "reward": 0.0, "score": 0.0, "done": False})

    sys.modules["requests"] = types.SimpleNamespace(
        post=post,
        get=lambda *a, **k: _Resp({"look_around": "", "inventory": "",
                                   "get_valid_action_object_combinations": ""}))
    for name in ("tiktoken", "openai", "numpy"):
        sys.modules.setdefault(name, types.ModuleType(name))
    mmengine = types.ModuleType("mmengine")
    mmengine.load = lambda *a, **k: {}
    sys.modules["mmengine"] = mmengine
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
    sys.modules["transformers"] = transformers
    for name in ("fastchat", "fastchat.model", "fastchat.model.model_adapter"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["fastchat.model.model_adapter"].get_conversation_template = lambda _: None


def child(task):
    """Walk the dispatch for one task. TASK is read at import, hence a fresh interpreter."""
    sys.path.insert(0, str(ROOT))
    _install_stubs()
    import mcts_collection
    import eval as eval_mod

    url = "http://127.0.0.1:36001"
    client = mcts_collection.initialize_environment(task, url)
    eval_client = eval_mod.initialize_environment(task, url)
    if type(client) is not type(eval_client):
        raise SystemExit(f"collection gives {type(client).__name__} but eval gives "
                         f"{type(eval_client).__name__}")
    client.reset(0)
    action = "Action:\n```sql\nSELECT 1\n```" if task == "intercode_sql" else "Action: look"
    step = client.step(action)
    if step is None or not hasattr(step, "reward"):
        raise SystemExit("step() did not return a StepOutput")

    # ExtendedMCTS.load must return the reconstructed root. Dropping its final return is silent
    # until revise runs and path_collection does `root.value` on None - which is how the
    # InterCode-SQL smoke run failed on its third attempt.
    module = sys.modules[mcts_collection.ExtendedMCTS.__module__]
    module.mmengine.load = lambda _p: {
        "visits": 1, "value": 0.5, "prior": 1, "puct_value": 0.0, "obs": "", "llm_response": "ROOT",
        "depth": 0, "is_terminal": False, "recent_actions": [], "action": "ROOT", "env_score": 0,
        "disaster": False, "state": [], "children": [],
    }
    root = mcts_collection.ExtendedMCTS.load("ignored")
    if root is None or not hasattr(root, "value"):
        raise SystemExit("ExtendedMCTS.load returned None - its final `return dict_to_node(...)` "
                         "is missing; path_collection will crash on root.value")

    # env_action decides what reaches the environment. Splitting "Action:" off a reply breaks any
    # environment whose own parser needs that marker - which is how the InterCode-SQL smoke run
    # scored 0 on every task while looking healthy.
    from mcts_utils.llm_server import env_action
    reply = ("Thought: t\nAction: \n```sql\nSELECT 1\n```" if task == "intercode_sql"
             else "Thought: t\nAction: look")
    sent = env_action(reply)
    if task in ("intercode_sql", "webshop"):
        if sent != reply:
            raise SystemExit(f"env_action must pass the raw reply for {task} under eto; "
                             f"it sent {sent!r}")
    if task == "intercode_sql":
        # Their parser needs mysql.connector, which only the environment image has. Where it is
        # importable, use it; otherwise assert the one property that actually broke - the marker
        # the parser keys on must still be there.
        sys.path.insert(0, os.environ["INTERCODE_ETO_ROOT"])
        try:
            from eval_agent.intercode_sql_action import parse_sql_action
        except ImportError:
            import re
            if len(re.findall(r"(?m)^Action:", sent)) != 1:
                raise SystemExit("env_action stripped the Action: marker InterCode requires")
        else:
            parse_sql_action(sent)      # raises if the environment would refuse it

    print(f"{type(client).__name__} conversation_start={len(type(client).conversation_start)} "
          f"load_ok={root.value} env_action_ok=1")


# ---------------------------------------------------------------- parent
def main():
    if not Path(CO).exists():
        sys.exit(f"needs the Co-Evolving checkout at {CO} (set COEVOLVING_ROOT)")
    failures = 0
    print("task dispatch (collection + eval, eto protocols):")
    for task, extra, expected_client, expected_turns in CASES:
        env = dict(os.environ, TASK=task, MAX_DEPTH="4", ITERA="2", N_GEN="2",
                   MAX_TOKEN_LENGTH="8192", MODEL_DIR="/none", MODEL_TYPE="t", MODEL_NAME="m",
                   ALPHA="0.5", BETA="0.2", WEBSHOP_ETO_ROOT=CO, SCIWORLD_ETO_ROOT=CO,
                   INTERCODE_ETO_ROOT=CO, **extra)
        run = subprocess.run([sys.executable, str(Path(__file__)), "--child", task],
                             env=env, capture_output=True, text=True)
        out = run.stdout.strip()
        ok = run.returncode == 0 and expected_client in out
        if ok and expected_turns is not None:
            ok = f"conversation_start={expected_turns}" in out
        if not ok:
            failures += 1
            detail = out or run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "?"
        else:
            detail = out
        print(f"  {'ok  ' if ok else 'FAIL'} {task:14s} {detail}")
    print("PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        sys.exit(main())
